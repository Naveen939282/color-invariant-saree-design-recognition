# Public Dataset Audit Findings

The supplied archive is the public Roboflow export “Indian Fabric Patterns” (MIT, exported May 26, 2024). It contains **1,470 archive members**: 1,468 JPG images and two README files. The images are arranged into train/valid/test and four textile/pattern categories, and all are 640x640 RGB. The archive README states that three salt-and-pepper-noise versions were created for each source image.

## Measured Counts

| Split | Banarasi | Bandhani | Ikat | Pichwai | Total |
|---|---:|---:|---:|---:|---:|
| Train | 432 | 279 | 303 | 279 | 1,293 |
| Validation | 43 | 22 | 26 | 24 | 115 |
| Test | 14 | 15 | 13 | 18 | 60 |
| Total | 489 | 316 | 342 | 321 | 1,468 |

## Candidate Sources, Variants, and Leakage

Candidate source IDs remove the Roboflow `.rf.<hexhash>` suffix, normalize the filename's encoded source extension, and scope the result by category. The Roboflow hash is not used as identity. This produces 606 filename groups: 175 groups of one image, zero groups of two, 431 groups of three, and zero groups above three. The README's three-version statement is therefore not true of every filename group. There are 265 groups with generic `image`/`images` basenames, and 53 normalized filename keys occur across more than one category; class scoping prevents cross-category merges. Same-category filename collisions remain impossible to rule out from filenames alone. No candidate source group crosses train/validation/test.

All five exact MD5 duplicate groups occur within train (two files per group); none occur within validation or test, and no exact duplicate hash crosses a split. Candidate filename-group overlap is also zero for train/validation, train/test, and validation/test. These checks reduce evidence of split leakage but do not establish true source identity for singleton or colliding filenames.

Across the 431 multi-image groups, median mean absolute RGB difference from the first filename-ordered image is 1.97/255; 376 groups are at or below 3/255. The representative variant sheet shows close visual copies, consistent with low-amplitude noise variants. Similarity and filename grouping are evidence only, not identity ground truth.

## Objective Limitation and Next Experiment

Banarasi, Bandhani, Ikat, and Pichwai are category labels, **not individual design IDs**. This dataset cannot directly provide real-world same-design/different-color pairs, and no design IDs are inferred in this audit. Any use of these candidates must be reported as **controlled synthetic color-invariance evaluation**, not real cross-color evaluation. At the audit stage, no model training, checkpoint creation, or synthetic color transformation was performed; the later controlled experiment is documented separately in `training_report.md`.

The reproducible generated `metadata.csv`, `source_groups.csv`, full summary, and contact sheets are under the ignored `local_outputs/kaggle_audit/` directory.