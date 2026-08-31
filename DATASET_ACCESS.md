# Dataset Access Checklist

Start registrations for the top 4 **today** — some approvals take days, and everything downstream
depends on them. Track status in the table as you go.

## Tier 1 — required to start (register first)

| Dataset | Purpose | Portal | Status |
|---|---|---|---|
| BEAT2 / EMAGE | Main coupled train/val set | https://pantomatrix.github.io/EMAGE/ | [ ] |
| AMASS | Body temporal prior pretraining | https://amass.is.tue.mpg.de/ | [ ] |
| VOCASET | Facial motion pretraining | https://voca.is.tue.mpg.de/ | [ ] |
| Gaze360 | Gaze branch training/eval | http://gaze360.csail.mit.edu/ | [ ] |
| SMPL-X body model files | Required by the geometry generator | https://smpl-x.is.tue.mpg.de/ | [ ] |
| FLAME model files | Required by the geometry generator | https://flame.is.tue.mpg.de/ | [ ] |

## Tier 2 — needed by Week 3-4 (evaluation + initializer training)

| Dataset | Purpose | Portal | Status |
|---|---|---|---|
| 3DPW | Real-world body reconstruction eval | https://virtualhumans.mpi-inf.mpg.de/3DPW/ | [ ] |
| Human3.6M | Controlled MPJPE/PA-MPJPE eval | https://vision.imar.ro/human3.6m/ | [ ] |
| BEDLAM | Monocular initializer training | https://bedlam.is.tue.mpg.de/ | [ ] |

## Tier 3 — stress tests / cross-dataset validation (Week 10-11)

| Dataset | Purpose | Portal | Status |
|---|---|---|---|
| Hi4D | Occlusion / contact stress test | https://yifeiyin04.github.io/Hi4D/ | [ ] |
| HUMBI | Cross-view holistic validation | https://humbi-data.net/ | [ ] |
| TalkSHOW | External coupled sequence eval | https://talkshow.is.tue.mpg.de/ | [ ] |
| MPIIFaceGaze | Secondary gaze benchmark | DaRUS Dataverse (CC BY-NC-SA 4.0) | [ ] |

## Optional (only if scope expands)
BEAT (original), FaceWarehouse, HumanML3D, BABEL, TotalCapture, CMU Panoptic.

## Notes
- Use the **official portal** for every dataset, not third-party mirrors — you need the current
  license terms and correct metadata.
- Several require a `.edu`/institutional email or a signed research-use agreement — start these
  even before you've written any code.
- Log the exact license (research-only vs non-commercial vs CC) for each dataset you actually use;
  you'll need this for the reproducibility section of the paper.
