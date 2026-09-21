//! `sdp_prefill` must be causal even when the caller passes no mask.
//!
//! Model code no longer builds an explicit causal mask when a fused
//! attention backend is compiled in (the backend owns masking), but a fresh
//! prefill without block tables still falls back to `sdp_prefill`. If that
//! fallback does not mask on its own, every prompt token attends to future
//! tokens and generation degrades to garbage.

use attention_rs::{InputMetadata, PagedAttention};
use candle_core::{DType, Device, Result, Tensor, D};

const HEADS: usize = 4;
const KV_HEADS: usize = 2;
const HEAD_DIM: usize = 16;

fn prefill_metadata(seqlens: &[u32], device: &Device) -> Result<InputMetadata> {
    let mut cu = vec![0u32];
    for s in seqlens {
        cu.push(cu.last().unwrap() + s);
    }
    let total = *cu.last().unwrap() as usize;
    let max = *seqlens.iter().max().unwrap() as usize;
    let cu = Tensor::new(cu, device)?;
    Ok(InputMetadata {
        is_prefill: true,
        is_mla: false,
        sequence_ids: None,
        mamba_slot_mapping: None,
        slot_mapping: Tensor::zeros(total, DType::I64, device)?,
        block_tables: None,
        block_tables_host: None,
        context_lens_host: None,
        context_lens: None,
        cu_seqlens_q: Some(cu.clone()),
        cu_seqlens_k: Some(cu),
        max_seqlen_q: max,
        max_seqlen_k: max,
        max_context_len: max,
        seqlens: Some(seqlens.to_vec()),
        flashinfer_metadata: None,
        is_mtp_verify: false,
    })
}

/// Reference causal attention for one sequence in batch-major layout
/// `[1, heads, seq, dim]`, GQA-expanded the same way `sdp_prefill` does.
fn reference_causal(q: &Tensor, k: &Tensor, v: &Tensor, scale: f64) -> Result<Tensor> {
    let rep = HEADS / KV_HEADS;
    let (_, _, seq, _) = q.dims4()?;
    let k = Tensor::cat(&vec![k; rep], 2)?.reshape((1, HEADS, seq, HEAD_DIM))?;
    let v = Tensor::cat(&vec![v; rep], 2)?.reshape((1, HEADS, seq, HEAD_DIM))?;
    let att = (q.matmul(&k.t()?)? * scale)?;
    let mask: Vec<f32> = (0..seq)
        .flat_map(|i| (0..seq).map(move |j| if j > i { f32::NEG_INFINITY } else { 0.0 }))
        .collect();
    let mask = Tensor::from_vec(mask, (seq, seq), q.device())?;
    let att = candle_nn::ops::softmax_last_dim(&att.broadcast_add(&mask)?)?;
    att.matmul(&v)?.transpose(1, 2)
}

/// `mask::causal_mask` has no CPU kernel, so run on the build's GPU backend.
fn gpu_device() -> Result<Device> {
    #[cfg(feature = "cuda")]
    return Device::new_cuda(0);
    #[cfg(all(feature = "metal", not(feature = "cuda")))]
    return Device::new_metal(0);
    #[cfg(not(any(feature = "cuda", feature = "metal")))]
    candle_core::bail!("sdp_prefill_causal needs the cuda or metal feature")
}

#[test]
fn sdp_prefill_without_mask_is_causal() -> Result<()> {
    let device = gpu_device()?;
    let scale = 1.0 / (HEAD_DIM as f64).sqrt();
    let attn = PagedAttention::new(
        HEADS,
        HEAD_DIM,
        scale as f32,
        Some(KV_HEADS),
        None,
        device.clone(),
        None,
        false,
    )?;

    // Two packed sequences, so the mask must also restart per sequence.
    let seqlens = [5u32, 3u32];
    let total = 8;
    let q = Tensor::randn(0f32, 1f32, (1, HEADS, total, HEAD_DIM), &device)?;
    let k = Tensor::randn(0f32, 1f32, (1, KV_HEADS, total, HEAD_DIM), &device)?;
    let v = Tensor::randn(0f32, 1f32, (1, KV_HEADS, total, HEAD_DIM), &device)?;
    let meta = prefill_metadata(&seqlens, &device)?;

    let out = attn.sdp_prefill(&q, &k, &v, None, &meta, None)?; // [1, total, heads, dim]

    let mut start = 0;
    for &len in &seqlens {
        let len = len as usize;
        let expected = reference_causal(
            &q.narrow(2, start, len)?,
            &k.narrow(2, start, len)?,
            &v.narrow(2, start, len)?,
            scale,
        )?;
        let got = out.narrow(1, start, len)?;
        let max_diff = (got - &expected)?
            .abs()?
            .flatten_all()?
            .max(D::Minus1)?
            .to_scalar::<f32>()?;
        assert!(
            max_diff < 1e-4,
            "sequence at offset {start}: sdp_prefill without a mask differs from causal attention (max diff {max_diff})"
        );

        // The first token of each sequence can only see itself, so its output
        // is exactly its own value row (GQA: query head h reads kv head h / rep).
        let rep = HEADS / KV_HEADS;
        for h in 0..HEADS {
            let first = out.narrow(1, start, 1)?.narrow(2, h, 1)?.flatten_all()?;
            let own_v = v
                .narrow(1, h / rep, 1)?
                .narrow(2, start, 1)?
                .flatten_all()?;
            let diff = (first - own_v)?.abs()?.max(D::Minus1)?.to_scalar::<f32>()?;
            assert!(
                diff < 1e-5,
                "head {h}: first token attended past itself (diff {diff})"
            );
        }
        start += len;
    }
    Ok(())
}
