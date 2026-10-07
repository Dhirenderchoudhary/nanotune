---
"@nanocollective/nanotune": minor
---

Stop a training run when validation loss stops improving, and copy the best saved checkpoint over `adapters.safetensors`. Patience defaults to 0, so existing runs are unchanged. Closes #207.
