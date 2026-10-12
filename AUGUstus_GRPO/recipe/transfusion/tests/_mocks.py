"""Shared lightweight mocks for the recipe tests (extracted from the
previously-inlined helpers of the removed diffusion-RL test module)."""

import torch


class _MockDiffusion:
    timestep_shift = 1.0
    # Transfusion's checkpoint predicts x0: the adapter must dispatch to the
    # x0-parameterised Gaussian kernel (fm_sde_step_with_logprob_x0).
    prediction_type = "x0"

    def output_to_velocity(self, model_out, x_t, t):
        # model_out is the x0 prediction (prediction_type="x0").
        t_exp = t.view(-1, *([1] * (x_t.ndim - 1))).clamp(min=1e-5)
        return (x_t - model_out) / t_exp

    def q_sample(self, x_0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_0)
        t_exp = t.view(-1, *([1] * (x_0.ndim - 1)))
        return (1 - t_exp) * x_0 + t_exp * noise, noise

    def predict_x0(self, model_out, x_t, t):
        return model_out


class _MockTextOut:
    past_key_values = None


class _MockModel(torch.nn.Module):
    """Minimal flow-matching model exposing the methods the adapter calls."""

    diffusion_type = "flow_matching"
    vit_feature_enabled = False
    scene_cond_enabled = False
    state_cond_enabled = False
    action_repr = "absolute"

    def __init__(self, traj_token=8, action_dim=3, hidden=16):
        super().__init__()
        self.traj_token = traj_token
        self.action_dim = action_dim
        self.hidden = hidden
        self.diffusion = _MockDiffusion()
        # Learnable denoiser params so the loss has a gradient path.
        self.w = torch.nn.Parameter(torch.full((traj_token, action_dim), 0.3))
        self.b = torch.nn.Parameter(torch.zeros(traj_token, action_dim))

    # -- prefix building primitives (cheap stand-ins) --
    def _get_multimodal_embeddings(self, input_ids, *a, **k):
        B, P = input_ids.shape
        emb = torch.zeros(B, P, self.hidden)
        return (emb, None) if k.get("capture_vit") else emb

    def _make_causal_mask(self, P, B, device, dtype):
        return torch.zeros(B, 1, P, P, dtype=dtype)

    def _get_extended_position_ids(self, input_ids, *a, **k):
        B, P = input_ids.shape
        return torch.zeros(3, B, P, dtype=torch.long), None

    def _compute_adarms_cond(self, t, dtype):
        return None

    def _text_model(self, **kwargs):
        return _MockTextOut()

    def _next_position_ids(self, prefix_pos, n):
        B = prefix_pos.shape[1]
        return torch.zeros(3, B, n, dtype=torch.long)

    # -- differentiable per-step denoiser: returns an x0 prediction --
    def _denoise_step_cached(self, x_t, t, past_kv, prefix_len, B, device, dtype,
                             action_position_ids, cond_vecs=None):
        return x_t * self.w + self.b

    # Grad path (cache-free) twin of ``_denoise_step_cached`` — the real model
    # guarantees identical maths (see diffusion_adapter.prepare_context), so the
    # mock mirrors the same computation.
    def _denoise_fullseq_step(self, x_t, t, ctx_embeds, prefix_pos, attention_mask,
                              prefix_adarms_cond, cond_vecs=None):
        return x_t * self.w + self.b

    def sample_noise(self, shape, device):
        return torch.randn(*shape)

    def normalize_actions(self, a):
        return a

    def denormalize_actions(self, a):
        return a


def _mock_batch(B=4, P=5):
    return {
        "input_ids": torch.zeros(B, P, dtype=torch.long),
        "attention_mask": torch.ones(B, P, dtype=torch.long),
    }
