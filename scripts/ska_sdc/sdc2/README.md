# Science Data Challenge 2 Scoring API

The SKA Science Data Challenge #2 (https://sdc2.astronomers.skatelescope.org/) tasked participants with identifying and classifying sources in a synthetic radio data cube.

As with SDC1 comparing the catalogue of sources used to generate the data cube (the truth catalogue) with the 'submission catalogue' produced by a participant's solution will provide the means of determining the success of the solution.

## SDC2: Scoring a submission

Scoring a submission for SDC2 is provided through the `Sdc2Scorer` class. Typically, this is instantiated with two pandas.DataFrame objects (corresponding to the submission and truth catalogues). A method is also available to construct an `Sdc2Scorer` instance from the paths to the two catalogues.

Once `Sdc2Scorer` has been instantiated, its `run` method will run the scoring pipeline and evaluate a result.

The result is provided via a `Sdc2Score` object, which has properties providing feedback about the submission catalogue.

Below are several examples which should illustrate the API usage.

### Catalogue schema

Catalogue files should be in space-delimited ascii format. The column names must be present as the first row that is read, and should match the following:

```
id,
ra,
dec,
hi_size,
line_flux_integral,
central_freq,
pa,
i,
w20
```

Similarly, catalogue DataFrames should have column (Series) names that precisely match these. If this is not the case, the catalogue will fail validation.

### Example 1: Scoring catalogue files

```python
from ska_sdc import Sdc2Scorer

sub_cat_path = "/path/to/submission/catalogue.txt"
truth_cat_path = "/path/to/truth/catalogue.txt"

scorer = Sdc2Scorer.from_txt(
    sub_cat_path, truth_cat_path, sub_skiprows=0, truth_skiprows=0
)
scorer.run()
```

Note the optional skiprows keyword arguments which can be used to specify rows to be skipped when reading the file (e.g. header rows). For SDC2 the column names are inferred from the first row read, and should match the expected schema, as given above.

### Example 2: Scoring catalogue DataFrames

If the catalogues are already DataFrame objects, the scorer can be instantiated from these directly as follows:

```python
from ska_sdc import Sdc1Scorer

scorer = Sdc2Scorer(
    sub_cat_df,
    truth_cat_df,
)
scorer.run()
```

## Development

### Function prototypes

See the Sphinx documentation in `/docs`.

### Configuration

Configuration items, e.g. the expected column headers, should be kept in `ska_sdc/sdc2/conf/config.ini`. These are made available to the `Sdc2Scorer` instance through `self.config`.

### Framework

The scoring pipeline is broadly separated into four stages:

1. Data preparation (preprocessing)
2. Crossmatching
3. Data cleansing (postprocessing)
4. Score generation

#### Preprocessing

Catalogues can be modified prior to crossmatching through preprocessing.

To add a custom step to preprocessing, first create an empty stub by copying the `XMatchPreprocessingStepStub` class from `ska_sdc/sdc2/utils/xmatch_preprocessing_steps.py` and renaming it.

The class name then needs to be added to the `step_names` list in `ska_sdc.sdc2.sdc2_scorer.Sdc2Scorer` in order to be processed at runtime:

```python

cat_sub_prep = XMatchPreprocessing(
    step_names=["XMatchPreprocessingStepStub"]
).preprocess(cat=self.cat_sub)
```

The logic for the custom step should be placed in the `execute` function of the new class:

```python
def execute(self):
    """
    Execute the step.

    Returns:
        :class:`pandas.DataFrame`: Processed catalogue.
    """
    # Logic placeholder.
    return self.cat
```

Variables can be passed between the `Sdc2Scorer` instance and the new preprocessing function by adding keyword arguments to the preprocess function call e.g. in the above example, the submission catalogue, `self.cat_sub` is available as the class member `self.cat`.

#### Crossmatching

The crossmatching logic can be modified by adding a custom function to the `XMatch` class in `ska_sdc/sdc2/utils/xmatch.py`. A stub, `_stub`, illustrates an example that returns an empty dataframe.

The `func_name` parameter in `ska_sdc.sdc2.sdc2_scorer.Sdc2Scorer` then needs to be changed accordingly:

```python
cand_cat_sub = XMatch(cat_sub=cat_sub_prep, cat_truth=cat_truth_prep).execute(
    func_name="_stub"
)
```

Additonal variables can be passed between the `Sdc2Scorer` instance and the new crossmatching function by adding the keyword to the `XMatch` constructor e.g. in the above example, `cat_sub_prep` and `cat_truth_prep` will be available as the class members `self.cat_sub` and `self.cat_truth` respectively.

#### Postprocessing

The crossmatched catalogue can be modified through postprocessing.

To add a custom step to postprocessing, first create an empty stub by copying the `XMatchPostprocessingStepStub` class from `ska_sdc/sdc2/utils/xmatch_postprocessing_steps.py` and renaming it.

The class name then needs to be added to the `step_names` list in `ska_sdc.sdc2.sdc2_scorer.Sdc2Scorer` in order to be processed at runtime:

```python

cat_sub_prep = XMatchPostprocessing(
    step_names=["XMatchPostprocessingStepStub"]
).postprocess(cat=self.cat_sub)
```

The logic for the custom step should be placed in the `execute` function of the new class:

```python
def execute(self):
    """
    Execute the step.

    Returns:
        :class:`pandas.DataFrame`: Processed catalogue.
    """
    # Logic placeholder.
    return self.cat
```

Variables can be passed between the `Sdc2Scorer` instance and the new postprocessing function by adding keyword arguments to the postprocess function call e.g. in the above example, the submission catalogue, `self.cat_sub` is available as the class member `self.cat`.

#### Score generation

The logic to generate a score is in `ska_sdc/sdc2/utils/create_score.py`. It must return a populated instance of Sdc2Score.
