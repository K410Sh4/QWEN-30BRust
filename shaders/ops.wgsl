// Persistent activation/KV operations. FP32 preserves the reference path's precision.
struct Frame {
    hidden:u32, heads:u32, kvheads:u32, d:u32,
    ff:u32, top:u32, ctx:u32, pos:u32,
    eps:f32, theta:f32, experts:u32, vocab:u32,
};
@group(0) @binding(0) var<uniform> p:Frame;
@group(0) @binding(1) var<storage,read> a:array<f32>;
@group(0) @binding(2) var<storage,read> b:array<f32>;
@group(0) @binding(3) var<storage,read_write> out:array<f32>;
@group(0) @binding(4) var<storage,read_write> extra:array<f32>;
var<workgroup> reduction:array<f32,64>;

@compute @workgroup_size(64)
fn norm(@builtin(local_invocation_id) t:vec3<u32>) {
    var sum=0.0;for(var j=t.x;j<p.hidden;j+=64u){sum+=a[j]*a[j];}
    reduction[t.x]=sum;workgroupBarrier();
    for(var step=32u;step>0u;step/=2u){if t.x<step{reduction[t.x]+=reduction[t.x+step];}workgroupBarrier();}
    let inv=inverseSqrt(reduction[0]/f32(p.hidden)+p.eps);
    for(var j=t.x;j<p.hidden;j+=64u){out[j]=a[j]*inv*b[j];}
}
fn rotate_head(head:u32,lane:u32) {
    let base=head*p.d;var sum=0.0;
    for(var j=lane;j<p.d;j+=64u){sum+=a[base+j]*a[base+j];}
    reduction[lane]=sum;workgroupBarrier();
    for(var step=32u;step>0u;step/=2u){if lane<step{reduction[lane]+=reduction[lane+step];}workgroupBarrier();}
    let inv=inverseSqrt(reduction[0]/f32(p.d)+p.eps);
    for(var j=lane;j<p.d/2u;j+=64u){
        let angle=f32(p.pos)*pow(p.theta,-2.0*f32(j)/f32(p.d));
        let x=a[base+j]*inv*b[j];let y=a[base+j+p.d/2u]*inv*b[j+p.d/2u];
        out[base+j]=x*cos(angle)-y*sin(angle);
        out[base+j+p.d/2u]=x*sin(angle)+y*cos(angle);
    }
}
@compute @workgroup_size(64)
fn qrope(@builtin(workgroup_id) g:vec3<u32>,@builtin(local_invocation_id) t:vec3<u32>){rotate_head(g.x,t.x);}
@compute @workgroup_size(64)
fn krope(@builtin(workgroup_id) g:vec3<u32>,@builtin(local_invocation_id) t:vec3<u32>){rotate_head(g.x,t.x);}
@compute @workgroup_size(64)
fn store_kv(@builtin(global_invocation_id) g:vec3<u32>){
    let n=p.kvheads*p.d;if g.x<n{out[p.pos*n+g.x]=a[g.x];extra[p.pos*n+g.x]=b[g.x];}
}
@compute @workgroup_size(64)
fn scores(@builtin(global_invocation_id) g:vec3<u32>){
    let time=p.pos+1u;let head=g.x/time;let tick=g.x%time;if head>=p.heads{return;}
    let kh=head/(p.heads/p.kvheads);var dot=0.0;
    for(var j=0u;j<p.d;j++){dot+=a[head*p.d+j]*b[(tick*p.kvheads+kh)*p.d+j];}
    out[head*p.ctx+tick]=dot*inverseSqrt(f32(p.d));
}
@compute @workgroup_size(64)
fn normalize_scores(@builtin(workgroup_id) g:vec3<u32>,@builtin(local_invocation_id) t:vec3<u32>){
    let base=g.x*p.ctx;var maximum=-3.402823466e+38;
    for(var tick=t.x;tick<=p.pos;tick+=64u){maximum=max(maximum,out[base+tick]);}
    reduction[t.x]=maximum;workgroupBarrier();
    for(var step=32u;step>0u;step/=2u){if t.x<step{reduction[t.x]=max(reduction[t.x],reduction[t.x+step]);}workgroupBarrier();}
    let peak=reduction[0];workgroupBarrier();var sum=0.0;
    for(var tick=t.x;tick<=p.pos;tick+=64u){let v=exp(out[base+tick]-peak);out[base+tick]=v;sum+=v;}
    reduction[t.x]=sum;workgroupBarrier();
    for(var step=32u;step>0u;step/=2u){if t.x<step{reduction[t.x]+=reduction[t.x+step];}workgroupBarrier();}
    let total=reduction[0];
    for(var tick=t.x;tick<=p.pos;tick+=64u){out[base+tick]/=total;}
}
@compute @workgroup_size(64)
fn attention_values(@builtin(global_invocation_id) g:vec3<u32>){
    if g.x>=p.heads*p.d{return;}
    let head=g.x/p.d;let j=g.x%p.d;let kh=head/(p.heads/p.kvheads);var sum=0.0;
    for(var tick=0u;tick<=p.pos;tick++){sum+=a[head*p.ctx+tick]*b[(tick*p.kvheads+kh)*p.d+j];}
    out[g.x]=sum;
}
@compute @workgroup_size(64)
fn residual(@builtin(global_invocation_id) g:vec3<u32>){if g.x<p.hidden{out[g.x]+=a[g.x];}}
@compute @workgroup_size(64)
fn swiglu(@builtin(global_invocation_id) g:vec3<u32>){
    if g.x<p.ff*p.top{let x=a[g.x];out[g.x]=(x/(1.0+exp(-x)))*b[g.x];}
}
@compute @workgroup_size(64)
fn mix(@builtin(global_invocation_id) g:vec3<u32>){
    if g.x<p.hidden{var sum=0.0;for(var e=0u;e<p.top;e++){sum+=b[e]*a[e*p.hidden+g.x];}out[g.x]+=sum;}
}
