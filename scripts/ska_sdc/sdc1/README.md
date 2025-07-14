# Science Data Challenge 1 Scoring API

The SKA Science Data Challenge #1 (https://astronomers.skatelescope.org/ska-science-data-challenge-1/) tasked participants with identifying and classifying sources in synthetic radio images.

In addition to the synthetic images, participants were provided with a section of the 'truth catalogue' of sources used to generate the artificial data. Comparing the truth catalogue with the 'submission catalogue' produced by a participant's solution would provide a means of determining the success of the solution.

To evaluate the accuracy of the results, a program was developed to cross-match sources between the submission and truth catalogues, and calculate a 'score' based on the result of this cross-match.

This is an open-source implementation of the program used to score and rank the submissions for the first SKA Science Data Challenge (SDC). A number of improvements have been made, most notably the use of a more performant cross-match algorithm. As such it is not possible to make a direct comparison between the scores produced by this package and the original program. The original IDL code is available at: https://astronomers.skatelescope.org/ska-science-data-challenge-1/

## SDC1: Scoring a submission

Scoring a submission for SDC1 is provided through the `Sdc1Scorer` class. Typically, this is instantiated with two pandas.DataFrame objects (corresponding to the submission and truth catalogues), and the corresponding image frequency (560, 1400 or 9200 MHz). A method is also available to construct an `SdcScorer` instance from the paths to the two catalogues.

Once the `SdcScorer` has been instantiated, its `run` method will run the scoring pipeline and evaluate a result.

The result is provided via a `Sdc1Score` object, which has properties providing feedback about the submission catalogue.

Below are several examples which should illustrate the API useage.

### Catalogue schema

The catalogues must conform to the schema specified in the competition rules; as a summary, the expected columns for the DataFrames are:

```python
CAT_COLUMNS = [
    "id",
    "ra_core",
    "dec_core",
    "ra_cent",
    "dec_cent",
    "flux",
    "core_frac",
    "b_maj",
    "b_min",
    "pa",
    "size",
    "class",
]
```

These are the column names which will be applied if reading the catalogues from a file. Catalogue files should be space-delimited tables, and header rows should be ignored using the flags in the `Sdc1Scorer.from_txt()` method.

### Example 1: Scoring catalogue files

```python
from ska_sdc import Sdc1Scorer

sub_cat_path = "/path/to/submission/catalogue.txt"
truth_cat_path = "/path/to/truth/catalogue.txt"

scorer = Sdc1Scorer.from_txt(
    sub_cat_path,
    truth_cat_path,
    freq=1400,
    sub_skiprows=1,
    truth_skiprows=0
)
scorer.run()
print("Final score: {}".format(scorer.score.value))
```

Note the optional skiprows keyword arguments which can be used to specify rows to be skipped when reading the file (e.g. header rows)

### Example 2: Scoring catalogue DataFrames

If the catalogues are already DataFrame objects, the scorer can be instantiated from these directly as follows:

```python
from ska_sdc import Sdc1Scorer

scorer = Sdc1Scorer(
    sub_cat_df,
    truth_cat_df,
    freq=1400,
)
scorer.run()
```

### Example 3: Using Sdc1Scorer.run optional arguments

```python
scorer.run(
    mode=0, # 0, 1 for core, centroid position modes respectively
    train=False, # True to score based on training area only, else exclude
    detail=False, # True to return per-source scores and match catalogue
)
```

### Sdc1Score properties

The Sdc1Score object has a range of properties in addition to the score value, as follows:

- `value`: numerical score value (`score_det - n_false`)
- `n_det`: total number of detected sources in submission
- `n_match`: number of detections that were matched to truth sources
- `n_bad`: number of matched detections that failed to meet acceptance threshold
- `n_false`: number of detections that were not matched to truth sources
- `score_det`: total score for all matched sources
- `acc_pc`: accuracy percentage for matched sources
- `scores_df`: DataFrame of individual source scores for each property
- `match_df`: DataFrame of matched sources with corresponding truth sources

## SDC1 scoring pipeline description

This is a brief overview of the stages of the scoring pipeline.

### Stage 1: prep

- A new column corresponding to the _log(flux)_ is created for each catalogue dataframe.
- The area corresponding to the training dataset is removed from each catalogue, unless the `train=True` is passed to `Sdc1Scorer.run`, in which case only the training area will be selected.
- Additional features required by the catalogue cross-match step are calculated. The first such feature is the primary beam correction factor, which accounts for off-axis sources being apparently fainter than sources closer to the beam centre. In addition to this, the convolved size property estimates the apparent detected source size; this is significant for small/point-like sources where otherwise the small positional error could mean matches are spuriously ignored.

### Stage 2: crossmatch

- A positional crossmatch is performed using a k-dimensional tree space partitioning structure. All truth catalogue sources within a radius of each submitted source's convolved size are identified as candidate matches.

### Stage 3: sieve

- For each source's candidate matches, select the best by considering the difference in flux and source size.

### Stage 4: create_score

- Reject (but count) all matches that lie more than 5 sigma from the corresponding truth source (when considering position, flux and size).
- For each matched source, calculate the accuracy of the measured properties and from these to generate a total score. Each matched source can contribute up to a score of 1.0 to the total score. Penalise for incorrectly identified sources, by subtracting the number of unmatched sources from the total score. This yields the final score.
