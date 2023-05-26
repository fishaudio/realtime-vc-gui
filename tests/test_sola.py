import numpy as np
import torch
import torch.nn.functional as F
from scipy.signal import convolve


def test_numpy_sola():
    fade_frames = 90 * 44100 // 1000
    sola_search_frames = 12 * 44100 // 1000

    sola_buffer = np.random.randn(fade_frames).astype(np.float32)
    infer_wav = np.random.randn(fade_frames * 5).astype(np.float32)
    infer_wav[114 : 114 + fade_frames] = sola_buffer

    conv_input = infer_wav[None, : fade_frames + sola_search_frames]
    cor_nom = F.conv1d(
        torch.from_numpy(conv_input).float(),
        torch.from_numpy(sola_buffer[None, None, :]).float(),
    )

    cor_nom1 = convolve(conv_input, np.flip(sola_buffer[None, :]), mode="valid")
    assert np.allclose(cor_nom.numpy(), cor_nom1, atol=1e-2)

    cor_den = torch.sqrt(
        F.conv1d(
            torch.from_numpy(conv_input).float() ** 2,
            torch.ones(1, 1, fade_frames),
        )
        + 1e-8
    )

    cor_den1 = np.sqrt(
        convolve(conv_input**2, np.ones((1, fade_frames)), mode="valid") + 1e-8
    )
    assert np.allclose(cor_den.numpy(), cor_den1, atol=1e-2)

    sola_offset = torch.argmax(cor_nom[0] / cor_den[0])
    sola_offset1 = np.argmax(cor_nom1[0] / cor_den1[0])

    assert sola_offset == sola_offset1 == 114
