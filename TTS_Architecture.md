# Multilingual Text-to-Speech System: Architecture, Training, and Inference

## 1. Introduction

This document presents the architecture, training methodology, dataset plan, and inference workflow for a multilingual text-to-speech (TTS) system. The system is designed to run within a single-GPU (RTX 3060, 12GB VRAM) budget while covering multiple languages and producing natural, non-robotic speech.

**Design goals:**
- Broad multilingual coverage (6-10 languages, multiple language families)
- Natural-sounding speech (not flat/robotic)
- Parameter-efficient — trainable and runnable on a single consumer GPU
- The generative model is trained from scratch, not fine-tuned from an existing TTS checkpoint

## 2. Proposed Architecture

The system uses a **duration-based, non-autoregressive architecture** (in the VITS family) rather than a diffusion or large autoregressive transformer, for the reasons discussed in Section 6.

A pretrained **multilingual speech encoder is used only as a frozen feature extractor during training** — it supplies rich multilingual acoustic-linguistic targets without needing to be trained ourselves, which is what makes broad language coverage feasible on limited data. It is not used at inference.

```
                         ┌───────────────────────┐
                         │   Text + Language ID   │
                         └───────────┬───────────┘
                                     │
                         ┌───────────▼───────────┐
                         │      Phonemizer        │   off-the-shelf
                         │   (e.g. espeak-ng)      │   (not trained)
                         └───────────┬───────────┘
                                     │
        ┌────────────────────────────┴────────────────────────────┐
        │                    TRAINABLE CORE                        │
        │                                                            │
        │   ┌─────────────────────┐                                │
        │   │  Duration Predictor  │  stochastic, natural timing    │
        │   └──────────┬──────────┘                                │
        │              │                                            │
        │   ┌──────────▼──────────┐                                │
        │   │       Decoder         │  phonemes+duration → features │
        │   └──────────┬──────────┘                                │
        │              │                                            │
        │   ┌──────────▼──────────┐                                │
        │   │   Vocoder (+ Disc.)   │  features → waveform          │
        │   └──────────┬──────────┘                                │
        └──────────────┼────────────────────────────────────────────┘
                        │
                        ▼
                 Audio Waveform Output
```

**Component summary:**

| Component | Trained? | Role |
|---|---|---|
| Phonemizer | No (off-the-shelf) | Text → phoneme symbols |
| Multilingual speech encoder | No (frozen, training-time only) | Provides rich acoustic-linguistic targets during training |
| Duration predictor | Yes, from scratch | Predicts natural phoneme timing |
| Decoder | Yes, from scratch | Phonemes + duration → acoustic features |
| Vocoder + discriminator | Yes, from scratch (adversarial) | Acoustic features → waveform |

**Multilingual support:** a learned language-ID embedding conditions the duration predictor and decoder, so one shared model serves all languages instead of requiring a separate model per language.

**Naturalness:** the duration predictor is *stochastic* (samples natural timing variation rather than one fixed value per phoneme), and the decoder/vocoder are trained adversarially, which pushes generated audio toward realistic detail instead of a blurred statistical average — the two main levers against robotic-sounding output.

## 3. Training Process

```
 Multilingual Dataset (audio + text)
              │
              ▼
 ┌─────────────────────────────┐
 │ Frozen tools (not trained)    │
 │ Phonemizer + speech encoder    │──▶ cached phoneme + feature targets
 └─────────────────────────────┘
              │
              ▼
 ┌─────────────────────────────┐
 │  Duration Predictor            │◀─┐
 └──────────────┬─────────────┘   │
                │                    │  Loss + backpropagation
                ▼                    │  (adversarial signal flows
 ┌─────────────────────────────┐   │   back through all trainable
 │  Decoder                        │◀─┤   stages every step)
 └──────────────┬─────────────┘   │
                │                    │
                ▼                    │
 ┌─────────────────────────────┐   │
 │  Vocoder + Discriminator        │──┘
 └─────────────────────────────┘
```

**Key training details:**
- **Teacher forcing**: the decoder is trained using *ground-truth* durations (from forced alignment), not the duration predictor's own guesses — this prevents early-training errors in one component from corrupting the other's learning signal.
- **Losses**: reconstruction loss (decoder output vs. real acoustic features), adversarial loss (discriminator, for realism), duration loss (predicted vs. ground-truth durations), and a variational/KL term if a VAE-style latent is used.
- **Forced alignment** provides the ground-truth phoneme durations required for training.
- Frozen stages contribute no gradients and are computed once, cached to disk, and reused — this keeps the trainable parameter count and training-time VRAM low enough for a single 3060.

## 4. Datasets

| Dataset | Coverage | Notes |
|---|---|---|
| **FLEURS** (Google) | ~102 languages, ~10 hrs each | Best source for genuine typological variety across language families |
| **Common Voice** (Mozilla) | 100+ languages | Crowd-sourced; large but variable quality |
| **CSS10** | 10 languages, single-speaker | Clean, purpose-built for small-scale multilingual TTS research |
| **VoxPopuli** | 15+ European languages | Clean, parliamentary recordings |
| **Multilingual LibriSpeech (MLS)** | 8 languages | High-quality audiobook data |

**Plan:** combine 6-10 languages across these sources, prioritizing typological diversity (not just raw language count) — e.g. including languages from Indo-European, Sino-Tibetan, and other families via FLEURS, rather than clustering only within one family.

## 5. Inference Workflow

```
        ┌─────────────────────┐
        │  Text + Language ID   │
        └──────────┬──────────┘
                   │
        ┌──────────▼──────────┐
        │      Phonemizer         │
        └──────────┬──────────┘
                   │
        ┌──────────▼──────────┐
        │  Duration Predictor    │   own prediction (no ground truth
        └──────────┬──────────┘   available at inference)
                   │
        ┌──────────▼──────────┐
        │       Decoder            │
        └──────────┬──────────┘
                   │
        ┌──────────▼──────────┐
        │       Vocoder            │
        └──────────┬──────────┘
                   │
                   ▼
            Audio Output
```

**Note:** the frozen multilingual speech encoder used during training does **not** appear at inference — it was needed only to help build training targets. Inference is therefore a lean, self-contained chain of just the trained modules plus the lightweight phonemizer, with no runtime dependency on any large pretrained model.

## 6. Why This Architecture, Not Diffusion or a Large Transformer

| Constraint | Diffusion (e.g. large diffusion-LM TTS) | Large autoregressive transformer (codec-LM) | Duration-based non-autoregressive (chosen) |
|---|---|---|---|
| Inference speed | Slow without extra distillation step | Slow (sequential token generation) | Fast — single forward pass |
| Data needed to converge well | High | High | Moderate |
| Training-time VRAM | High | High (long-sequence attention) | Lower |
| Single-GPU, from-scratch precedent | Rare at small scale | Prone to instability with limited data | Well established (basis of VITS) |

Given the hardware and data constraints, the duration-based non-autoregressive design is the only one of the three with a proven track record of converging to good quality from random initialization on a single consumer GPU.

## 7. Evaluation Plan

- **Quality**: automatic MOS-prediction tools (e.g. UTMOS) on generated samples
- **Intelligibility**: transcribe generated audio with Whisper, compute Word Error Rate against source text
- **Efficiency**: report trainable parameter count and single-GPU training time as the core efficiency metric
- **Comparison framing**: quality achieved relative to a large reference system (e.g. OmniVoice), normalized by parameter count and compute budget — not a raw quality claim

## 8. Conclusion

This architecture is designed around the project's actual constraints: a single 12GB GPU, limited per-language data, and a requirement that the generative model be trained from scratch. By using a frozen multilingual speech encoder purely as a training-time feature extractor, and adopting a duration-based non-autoregressive generative design, the system targets broad multilingual coverage and natural-sounding speech within a realistic, defensible compute and data budget.
