# Third-Party Components

`vendor/mae_st` is copied unchanged from the existing study's pinned MAE-ST
source. Its license, source manifest and upstream attribution are included.

TrackMAE teacher inference uses [CoTracker](https://github.com/facebookresearch/co-tracker).
Its source/weights are external and governed by the upstream terms. This
package does not redistribute these weights or download them during training.

The shared-backbone CMR method adapters and revised cineQMAE query/loss kernels
are reused from the user's study. No new pretrained initialization is silently
introduced. Baseline adapters are not claimed to be official full reproductions.
