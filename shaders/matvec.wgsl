// Quantized weights remain packed in GPU memory. No shader-f16 requirement.
struct Params { cols:u32, rows:u32, kind:u32, row_bytes:u32, out_offset:u32, batch_bytes:u32, input_stride:u32, pad:u32 };
@group(0) @binding(0) var<storage,read> weights:array<u32>;
@group(0) @binding(1) var<storage,read> input:array<f32>;
@group(0) @binding(2) var<storage,read_write> output:array<f32>;
@group(0) @binding(3) var<uniform> p:Params;
var<workgroup> sums:array<f32,64>;
fn byte_at(n:u32)->u32 { return (weights[n/4u] >> ((n%4u)*8u)) & 255u; }
fn signed_byte(n:u32)->f32 { let v=byte_at(n); return f32(i32(v)-select(0,256,v>=128u)); }
fn half_at(n:u32)->f32 { return unpack2x16float(byte_at(n) | (byte_at(n+1u)<<8u)).x; }
fn scale_at(base:u32,j:u32)->vec2<f32> {
    if j<4u { return vec2<f32>(f32(byte_at(base+j)&63u),f32(byte_at(base+j+4u)&63u)); }
    return vec2<f32>(f32((byte_at(base+j+4u)&15u)|((byte_at(base+j-4u)>>6u)<<4u)),
        f32((byte_at(base+j+4u)>>4u)|((byte_at(base+j)>>6u)<<4u)));
}
fn weight(row:u32,col:u32,batch:u32)->f32 {
    let r=batch*p.batch_bytes+row*p.row_bytes;
    if p.kind==0u {return bitcast<f32>(weights[(r+col*4u)/4u]);}
    if p.kind==1u {return half_at(r+col*2u);}
    if p.kind==2u {
        let b=r+(col/32u)*18u;let i=col%32u;
        let q=(byte_at(b+2u+i%16u)>>select(4u,0u,i<16u))&15u;
        return half_at(b)*f32(i32(q)-8);
    }
    if p.kind==8u {let b=r+(col/32u)*34u;return half_at(b)*signed_byte(b+2u+col%32u);}
    if p.kind==12u || p.kind==13u {
        let b=r+(col/256u)*select(176u,144u,p.kind==12u);let i=col%256u;let g=i/32u;
        let sm=scale_at(b+4u,g);
        let qs=b+select(48u,16u,p.kind==12u);
        var q=(byte_at(qs+(i/64u)*32u+i%32u)>>((g%2u)*4u))&15u;
        if p.kind==13u && (byte_at(b+16u+i%32u)&(1u<<g))!=0u {q+=16u;}
        return half_at(b)*sm.x*f32(q)-half_at(b+2u)*sm.y;
    }
    // Q6_K: 128 low bits bytes, 64 high bits bytes, 16 signed scales, FP16 d.
    let b=r+(col/256u)*210u;let i=col%256u;let c=i/128u;let j=i%128u;let quarter=j/32u;let lane=j%32u;
    let low=byte_at(b+c*64u+(quarter%2u)*32u+lane);
    let lo=(low>>((quarter/2u)*4u))&15u;
    let hi=(byte_at(b+128u+c*32u+lane)>>(quarter*2u))&3u;
    let q=i32(lo|(hi<<4u))-32;
    return half_at(b+208u)*signed_byte(b+192u+c*8u+quarter*2u+lane/16u)*f32(q);
}
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) gid:vec3<u32>, @builtin(local_invocation_id) tid:vec3<u32>) {
    // 2D grid accommodates vocabularies larger than Vulkan's 65535 grid limit.
    let row=gid.x+gid.y*65535u;
    if row>=p.rows {return;}
    var sum=0.0;
    for(var col=tid.x;col<p.cols;col+=64u){sum+=weight(row,col,gid.z)*input[gid.z*p.input_stride+col];}
    sums[tid.x]=sum;workgroupBarrier();
    for(var stride=32u;stride>0u;stride/=2u){
        if tid.x<stride {sums[tid.x]+=sums[tid.x+stride];}workgroupBarrier();
    }
    if tid.x==0u {output[p.out_offset+gid.z*p.rows+row]=sums[0];}
}
