# Model Card — <Model Name> (v<version>)

## 1. Overview
- **Goal:** (what problem this model solves)
- **Task:** Object Detection
- **Classes:** (list classes detected)
- **Intended users:** (e.g., operators, QA engineers, developers)

## 2. Intended Use and Scope
- **Intended environment:** (factory bench / warehouse / etc.)
- **Assumptions:** (camera viewpoint, distance, lighting)
- **Out-of-scope uses:** (what it is not designed for)

## 3. Training Data
- **Data source:** (collected by group, how/where)
- **Dataset size:** total images and per split
- **Class distribution:** counts per class
- **Labeling guidelines:** key rules and edge cases
- **Augmentations:** what/why (Roboflow settings summary)

## 4. Evaluation
- **Test set description:** size, how separated from training
- **Metrics:** mAP@0.5, mAP@0.5:0.95, precision, recall (overall + per-class)
- **Qualitative analysis:** example successes and failures
- **Recommended confidence threshold(s):** and rationale

## 5. Limitations and Failure Modes
- Known failure conditions (glare, occlusion, tiny objects, etc.)
- Typical false positives / false negatives

## 6. Deployment Notes
- **Input requirements:** resolution, color space, preprocessing
- **Output format:** bounding box convention, score meaning
- **Compute:** approximate latency / hardware notes

## 7. Ethical / Safety / Privacy Considerations
- Any risks (faces, sensitive info, misuse)
- How you mitigated them (cropping, consent, etc.)

## 8. Versioning and Contact
- **Version:** vX.Y
- **Date:**
- **Authors:**
