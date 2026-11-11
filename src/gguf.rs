use crate::quant;
use anyhow::{bail, ensure, Context, Result};
use memmap2::Mmap;
use serde_json::Value;
use std::{
    collections::BTreeMap,
    fs::File,
    io::{BufReader, Read, Seek},
    path::Path,
    sync::atomic::{AtomicUsize, Ordering},
};
static NEXT_MODEL_ID: AtomicUsize = AtomicUsize::new(1);

#[derive(Debug, Clone)]
pub struct Tensor {
    pub name: String,
    pub dims: Vec<usize>,
    pub kind: u32,
    pub offset: usize,
    pub len: usize,
}
pub struct Gguf {
    pub metadata: BTreeMap<String, Value>,
    pub tensors: BTreeMap<String, Tensor>,
    mmap: Mmap,
    pub(crate) id: usize,
}
struct Reader {
    r: BufReader<File>,
    budget: usize,
}
impl Reader {
    fn bytes<const N: usize>(&mut self) -> Result<[u8; N]> {
        let mut b = [0; N];
        self.r.read_exact(&mut b)?;
        Ok(b)
    }
    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.bytes()?))
    }
    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.bytes()?))
    }
    fn count(&mut self, max: usize) -> Result<usize> {
        let n = usize::try_from(self.u64()?)?;
        ensure!(n <= max, "GGUF count {n} exceeds limit {max}");
        Ok(n)
    }
    fn string(&mut self) -> Result<String> {
        let n = self.count(16 * 1024 * 1024)?;
        ensure!(n <= self.budget, "GGUF metadata budget exceeded");
        self.budget -= n;
        let mut b = vec![0; n];
        self.r.read_exact(&mut b)?;
        Ok(String::from_utf8(b)?)
    }
    fn val(&mut self, t: u32, depth: usize) -> Result<Value> {
        ensure!(
            depth <= 4 && self.budget >= 8,
            "excessive metadata depth/size"
        );
        self.budget -= 8;
        Ok(match t {
            0 => self.bytes::<1>()?[0].into(),
            1 => (self.bytes::<1>()?[0] as i8).into(),
            2 => u16::from_le_bytes(self.bytes()?).into(),
            3 => i16::from_le_bytes(self.bytes()?).into(),
            4 => self.u32()?.into(),
            5 => i32::from_le_bytes(self.bytes()?).into(),
            6 => {
                let f = f32::from_le_bytes(self.bytes()?);
                ensure!(f.is_finite(), "nonfinite metadata");
                Value::from(f)
            }
            7 => {
                let x = self.bytes::<1>()?[0];
                ensure!(x <= 1, "invalid boolean");
                (x != 0).into()
            }
            8 => self.string()?.into(),
            9 => {
                let ty = self.u32()?;
                let n = self.count(1_000_000)?;
                let mut v = Vec::with_capacity(n);
                for _ in 0..n {
                    v.push(self.val(ty, depth + 1)?);
                }
                Value::Array(v)
            }
            10 => self.u64()?.into(),
            11 => i64::from_le_bytes(self.bytes()?).into(),
            12 => {
                let f = f64::from_le_bytes(self.bytes()?);
                ensure!(f.is_finite(), "nonfinite metadata");
                Value::from(f)
            }
            _ => bail!("unknown GGUF metadata type {t}"),
        })
    }
}
impl Gguf {
    pub fn open(path: &Path) -> Result<Self> {
        let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
        let file_len = usize::try_from(file.metadata()?.len())?;
        let mut r = Reader {
            r: BufReader::new(file.try_clone()?),
            budget: 128 * 1024 * 1024,
        };
        ensure!(&r.bytes::<4>()? == b"GGUF", "not a GGUF file");
        let ver = r.u32()?;
        ensure!(ver == 2 || ver == 3, "GGUF version {ver} unsupported");
        let nt = r.count(100_000)?;
        let nk = r.count(100_000)?;
        let mut metadata = BTreeMap::new();
        for _ in 0..nk {
            let k = r.string()?;
            let t = r.u32()?;
            let v = r.val(t, 0)?;
            ensure!(metadata.insert(k, v).is_none(), "duplicate metadata key");
        }
        ensure!(
            metadata
                .get("split.count")
                .and_then(Value::as_u64)
                .unwrap_or(1)
                == 1,
            "split GGUF not supported; merge shards first"
        );
        let mut tensors = BTreeMap::new();
        for _ in 0..nt {
            let name = r.string()?;
            let nd = r.u32()? as usize;
            ensure!(
                (1..=3).contains(&nd),
                "unsupported tensor rank {nd}: {name}"
            );
            let mut dims = Vec::new();
            for _ in 0..nd {
                let d = usize::try_from(r.u64()?)?;
                ensure!(d > 0, "zero tensor dimension");
                dims.push(d);
            }
            let kind = r.u32()?;
            let offset = usize::try_from(r.u64()?)?;
            let len = dims[1..]
                .iter()
                .try_fold(quant::row_bytes(kind, dims[0])?, |a, &d| {
                    a.checked_mul(d).context("tensor length overflow")
                })?;
            let tensor = Tensor {
                name: name.clone(),
                dims,
                kind,
                offset,
                len,
            };
            ensure!(
                tensors.insert(name, tensor).is_none(),
                "duplicate tensor name"
            );
        }
        let alignment = metadata
            .get("general.alignment")
            .and_then(Value::as_u64)
            .unwrap_or(32) as usize;
        ensure!(
            alignment.is_power_of_two() && alignment <= 4096,
            "invalid GGUF alignment"
        );
        let p = usize::try_from(r.r.stream_position()?)?;
        let start = p.checked_add(alignment - 1).context("offset overflow")? & !(alignment - 1);
        let mut ranges = Vec::new();
        for t in tensors.values_mut() {
            ensure!(t.offset % alignment == 0, "unaligned tensor {}", t.name);
            t.offset = t
                .offset
                .checked_add(start)
                .context("tensor offset overflow")?;
            let end = t.offset.checked_add(t.len).context("tensor end overflow")?;
            ensure!(end <= file_len, "truncated tensor {}", t.name);
            ranges.push((t.offset, end));
        }
        ranges.sort_unstable();
        for pair in ranges.windows(2) {
            ensure!(pair[0].1 <= pair[1].0, "overlapping tensors");
        }
        // The model file must remain immutable throughout this mapping's lifetime.
        let mmap = unsafe { memmap2::MmapOptions::new().map(&file)? };
        Ok(Self {
            metadata,
            tensors,
            mmap,
            id: NEXT_MODEL_ID.fetch_add(1, Ordering::Relaxed),
        })
    }
    pub fn tensor(&self, name: &str) -> Result<&Tensor> {
        self.tensors
            .get(name)
            .with_context(|| format!("missing tensor {name}"))
    }
    pub fn data(&self, t: &Tensor) -> &[u8] {
        &self.mmap[t.offset..t.offset + t.len]
    }
    pub fn row(&self, name: &str, row: usize) -> Result<Vec<f32>> {
        let t = self.tensor(name)?;
        let size = quant::row_bytes(t.kind, t.dims[0])?;
        ensure!(row < t.len / size, "row out of bounds: {name}");
        quant::decode(
            &self.data(t)[row * size..(row + 1) * size],
            t.kind,
            t.dims[0],
        )
    }
    pub fn uint(&self, key: &str) -> Result<usize> {
        usize::try_from(
            self.metadata
                .get(key)
                .and_then(Value::as_u64)
                .with_context(|| format!("missing integer {key}"))?,
        )
        .map_err(Into::into)
    }
    pub fn float(&self, key: &str, default: f32) -> f32 {
        self.metadata
            .get(key)
            .and_then(Value::as_f64)
            .map(|x| x as f32)
            .unwrap_or(default)
    }
}
