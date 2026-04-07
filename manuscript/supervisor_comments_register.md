# Supervisor Comment Register

**Source**: `Thesis_31.01.2026_PV.pdf` (pvelasquez annotations)  
**Total comments**: 206  
**Workflow**: Check off each item as you address it in the new draft.

---

## Citations Missing (14 comments)

*Add explicit citations wherever claims are made without a reference. Supervisor flags ~15 specific spots.*

- [ ] **p21** — on: *«geographic focus for several reasons. First, the continent contains some of the world’s most biodive»*  
  > citation?

- [ ] **p21** — on: *«conservation efforts. Second, South America has experienced substantial expansion of pro- tected are»*  
  > citation?

- [ ] **p21** — on: *«of protected area establishment. Third, the region shows pronounced heterogeneity in eco- logical co»*  
  > citation?

- [ ] **p22** — on: *«The reference grid is defined in the coordinate reference system EPSG:3857 (WGS 84 / Web Mercator). »*  
  > citation

- [ ] **p22** — on: *«distortions»*  
  > how the distorsion happens? and why at high latitudes?  if you do not want to explain, please cite this information otherwise it sounds like you finds this out here.

- [ ] **p27** — on: *«path dependence and clustering effects,»*  
  > is this described somewhere in the manuscript? or cited?

- [ ] **p33** — on: *«Conceptually, the modelling framework is closely related to discrete-time hazard (or du- ration) mod»*  
  > citation

- [ ] **p40** — on: *«scale pos weight»*  
  > where is this used? software? citation? same of other model parameters.

- [ ] **p40** — on: *«F1-scor»*  
  > what is this? citation?

- [ ] **p40** — on: *«are uninformative and potentially misleading.»*  
  > who says this? citation?

- [ ] **p40** — on: *«) are robust to severe class imbalance»*  
  > citation

- [ ] **p40** — on: *«directly reflect the ranking quality of predicted risks,»*  
  > citation

- [ ] **p41** — on: *«Calibration curves and Brier scores»*  
  > who else uses it? what are they for? citation?

- [ ] **p44** — on: *«The»*  
  > use citations everythere here.


## Structure / Section Placement (12 comments)

*~20 comments about things being in the wrong section. Data vs. Methods distinction is a major issue. Terms defined after first use.*

- [ ] **p22** — on: *«This thesis integrates a diverse set of geospatial datasets to construct both the target variable (p»*  
  > this is part of your modelling framework not the description of your data

- [ ] **p23** — on: *«America. Unless stated otherwise, datasets are processed to match the 1 km × 1 km reference grid des»*  
  > this comes a bit late

- [ ] **p26** — on: *«The target variable indicates whether a grid cell transitions from an unprotected to a pro- tected s»*  
  > this is part of your method, right? how you use the data.

- [ ] **p26** — on: *«3.2.2 Target Variable: New Protected-Area Designations»*  
  > I think this section does not describe the data anymore. It shows what is done with the data and how things were defined. It looks like part of a method. I would label it differently to avoid any major re-structuring

- [ ] **p26** — on: *«. P»*  
  > what if a protected area turns into an unprotected area due to whatever reason?   Is this part of the model? if yes, considering the already protected areas would be mandatory, right?   If not part of the model, why?

- [ ] **p27** — on: *«3.2.3 Predictor Variables»*  
  > I think that defining predictor variables are part of the modelling framework?   Since you are not describing the data. It is more about how it will be used, right?

- [ ] **p32** — on: *«processes that extend beyond individual grid cells. This multi-scale representation allows the model»*  
  > this needs to move upward.

- [ ] **p32** — on: *«sion adjacent to existing reserves. By encoding proximity explicitly, distance-based measures comple»*  
  > this might moved slightly upward. it is rather a broader information and not in detail (e.g., explaining how it works: second and third sentences)

- [ ] **p34** — on: *«and a LightGBM transition model. The LightGBM model serves as the primary production model for forec»*  
  > this does not belong here.   Please say that you will describe then in the following .....

- [ ] **p36** — on: *«is implemented for Colombia (Model C). This pipeline mirrors the complete data processing, feature e»*  
  > You have not described them yet. Please re organise your manuscript accordingly.

- [ ] **p39** — on: *«Lookahead window definition»*  
  > this definition comes too late. it is used many times before!

- [ ] **p40** — on: *«Transitions into protected status are rare events,»*  
  > This defition comes TOO late!   Please considering re-structuring the manuscript, since there are many definition/terms used before being defined.


## Terms Used Before Being Defined (6 comments)

*~13 instances where a term is used several pages before it is defined (e.g., 'lookahead window', 'risk set', 'rare event').*

- [ ] **p22** — on: *«key predictors»*  
  > which key predictors.   is "predictor" already introduced in the manuscript?

- [ ] **p31** — on: *«predictors.»*  
  > this concept has not been yet introduced. (I know what it means) it could be that the reader understand it as part or a regression (econometric)

- [ ] **p31** — on: *«pixel in a given year, conditional on not yet being protected. They form the core explanatory variab»*  
  > explanatory variable has. not been introduced and why core?  are there other ones?

- [ ] **p31** — on: *«dynamic predictors.»*  
  > a new concept that has not been introduced yet.

- [ ] **p33** — on: *«PA»*  
  > I know the meaning. but was it already defined?   Keep in mind to make reader's life easy (always!), that means that one could define it again, assuming that the reader might forget it at some point

- [ ] **p37** — on: *«policy»*  
  > this was not introduced yet, what is meant by this?


## Clarity / Need Examples or Simpler Language (42 comments)

*The single biggest category (~42 comments). Many things need examples, simpler language, or clearer phrasing — especially ML concepts for a non-ML reader.*

- [ ] **p21** — on: *«Focusing on a single continent ensures institutional and ecological coherence while main- taining su»*  
  > this statement is very strong. Is there anything that can support it? like an example? or literature? or similar statements? If not: you would need to explain your reasoning slightly more in detail.

- [ ] **p21** — on: *«supports»*  
  > what does it mean?   is the pixel-level dataset or modeling setting at 1 km? why 1 km and not 2km or not 500m? is there any dataset that needs to be used as it is to decreases interpolation biases as much as possible?

- [ ] **p22** — on: *«reflects a trade-off»*  
  > What is meant by this?

- [ ] **p22** — on: *«and the native resolution of the majority of input datasets. While higher-resolution land- cover pro»*  
  > I think it is still unclear why 1 km.   is the landcover at 1 km?

- [ ] **p22** — on: *«Mercator). The reference grid is defined suitable for continental-scale spatial analysis in South Am»*  
  > what is meant with suitable?

- [ ] **p22** — on: *«distortions in area at high latitudes, these effects are limited within the latitudinal extent of So»*  
  > that needs to be clarified better.

- [ ] **p27** — on: *«climate variables, topographic characteristics such as elevation and slope, and global con- servatio»*  
  > example?

- [ ] **p28** — on: *«harmonised»*  
  > what is meant by this? any missing value is replaced by anything as well?

- [ ] **p28** — on: *«tions, such as incomplete temporal coverage in specific remote-sensing products. No global temporal »*  
  > i think i did not get this. please make it simpler

- [ ] **p29** — on: *«nearest-neighbour»*  
  > the nearest grid point? or using weighted 4  nearest grid points? please clarify it

- [ ] **p29** — on: *«temporal resolution and sampling frequency.»*  
  > is it not the same .... temporal resolution and sampling frequency. If not, it might need a short clarification

- [ ] **p29** — on: *«represented at the calendar-year level. Temporal handling occurs in two stages: within-year aggregat»*  
  > I think I could not understand it very much. Would you please make it a bit simpler or with examples?

- [ ] **p30** — on: *«annual raster»*  
  > what does it mean specifically?

- [ ] **p30** — on: *«within-year reduction.»*  
  > what does it mean? if it was defined before, I must have overlooked it, if so, why not a brief reminder in parenthesis?

- [ ] **p31** — on: *«only by conditions at a specific location but also by its surrounding landscape, the feature set com»*  
  > what is meant by this?

- [ ] **p31** — on: *«features»*  
  > what is meant by features?

- [ ] **p31** — on: *«event-history»*  
  > what is this?

- [ ] **p31** — on: *«scales.»*  
  > what kind of scales? temporal? spatial? industrial (small v/s big firms). What is meant by regional dynamics? what is the dynamic? and how does it influence the decisions?

- [ ] **p31** — on: *«uniform»*  
  > what is meant by this?

- [ ] **p33** — on: *«classification»*  
  > which classification?  unprotected to become protected?   what is the problem?  I do not get the problem formulation.   Please explain more what is meant by problem formulation and define the problem easier for the reader

- [ ] **p34** — on: *«extreme class imbalance»*  
  > what is meant by this?

- [ ] **p35** — on: *«class imbalance addressed using per-tree class weighting (c»*  
  > this needs to be explained in simpler ways as well.. it is very technical

- [ ] **p35** — on: *«Random Forests provide a flexible, nonparametric benchmark capable of capturing nonlin- earities and»*  
  > how? example?

- [ ] **p35** — on: *«Class imbalance»*  
  > what is meant by imbalance?

- [ ] **p35** — on: *«time-based sample weights,»*  
  > what is meant by this? what is the goal?

- [ ] **p35**  
  > aaah it comes later...   please rephrase it to make the reading easier

- [ ] **p35** — on: *«max depth, and min child samples) are selected using temporal cross-validation and op- timised for P»*  
  > same here. More details/explanation/easier/simpler

- [ ] **p35** — on: *«Final»*  
  > are the initial models? secondary models?  what is meant by final?

- [ ] **p35**  
  > this section needs further explanation and clarification

- [ ] **p36** — on: *«geneity in ecological conditions, land-use patterns, infrastructure development, and conser- vation »*  
  > what is this?  is it described previously?

- [ ] **p37** — on: *«include Platt scaling and, in robustness checks, isotonic regression. The fitted calibrator is appli»*  
  > very much technical. Please make it simpler and explain it further.

- [ ] **p37** — on: *«Because class-weighting schemes used to address extreme imbalance modify the effective loss function»*  
  > an example of this process would be helpful for the reader

- [ ] **p37** — on: *«policy»*  
  > this was not introduced yet, what is meant by this?

- [ ] **p37** — on: *«must preserve the temporal ordering of observations and avoid any leakage of future infor- mation in»*  
  > example?

- [ ] **p38** — on: *«includes PA establishments until 2024.»*  
  > I think I did not understand this, please clarify

- [ ] **p38** — on: *«2018–2019), while preserving the same five-year lookahead definition and right-censoring logic. Pred»*  
  > how? I think I am puzzled by the years used. Please clarify

- [ ] **p38** — on: *«To assess the sensitivity of model performance to the choice of evaluation window, alternative test »*  
  > this paragraph needs to be clearer. It triggers confusion at times. Giving examples might help. Using the same wording as previously might help too.

- [ ] **p39** — on: *«ping, using a fixed number of boosting iterations determined from temporal cross-validation, ensurin»*  
  > this needs more explanation using examples maybe

- [ ] **p40**  
  > I could not get it clear

- [ ] **p40**  
  > please give an example

- [ ] **p40** — on: *«F1-scor»*  
  > what is this? citation?

- [ ] **p42** — on: *«compromise»*  
  > I am puzzled by this term again... it sounds odd. What are the wins & losses that are balanced?


## Technical Depth / More Explanation Needed (12 comments)

*~12 comments asking for more detail: equations, tables of parameters, flowcharts, more explanation of calibration, hyperparameter tuning.*

- [ ] **p33** — on: *«classification»*  
  > which classification?  unprotected to become protected?   what is the problem?  I do not get the problem formulation.   Please explain more what is meant by problem formulation and define the problem easier for the reader

- [ ] **p34** — on: *«Static Suitability Baseline»*  
  > This needs better wording. The storytelling is a bit confusing.   what is is, (what for), what it does, more details, its results and how it's evaluated.  more or less this structure.

- [ ] **p34** — on: *«This study employs»*  
  > it would be good to have a table or list with the variables target or predictors, etc.

- [ ] **p35** — on: *«Hyperparameters governing tree structure and regularisation (including num leaves, max depth, and mi»*  
  > more explanation about this

- [ ] **p35** — on: *«max depth, and min child samples) are selected using temporal cross-validation and op- timised for P»*  
  > same here. More details/explanation/easier/simpler

- [ ] **p35**  
  > this section needs further explanation and clarification

- [ ] **p37**  
  > maybe a table?   same for other parameters.   Do you add any table or list of parameters that are tunned or can be adjusted ?   is there a chart flow somewhere that describes how everything works?

- [ ] **p37**  
  > you mention a decision tree.. is there any diagram of it?

- [ ] **p37** — on: *«early stopping and evaluated on the held-out test period.»*  
  > explain more

- [ ] **p37** — on: *«robustness checks, isotonic regression.»*  
  > explain more

- [ ] **p39** — on: *«ping, using a fixed number of boosting iterations determined from temporal cross-validation, ensurin»*  
  > this needs more explanation using examples maybe

- [ ] **p41** — on: *«primary evaluation metric in this study. In rare-event settings, PR–AUC is sensitive to performance »*  
  > You must explain them very much in detail since they are a key metric/outcome of your model performance. Please include equations if appropriate.   If the explanation is extremly arge (more than 3-4 pages, please add an appendix)


## Writing Style / Wording / Repetition (13 comments)

*~16 comments on repetitions, wording choices, line spacing, header length. Generally minor fixes.*

- [ ] **p3** — on: *«Literature Review»*  
  > Try to keep shorter headers throughout the manuscript

- [ ] **p21** — on: *«All analyses»*  
  > Which analysis? I think you can rephrase it avoiding using this word.

- [ ] **p22** — on: *«The use of EPSG:3857 ensures seamless integration with derived datasets and avoids repeated reprojec»*  
  > there are a few repetitions with the previous sentences. I suggest to rephrase them.

- [ ] **p23** — on: *«Table 1: Overview of datasets and variables used to predict protected area establishment.»*  
  > you could decrease the fontsize of the table if you want.

- [ ] **p25** — on: *«• High biodiversity areas • Intact wilderness areas • Climate stabilisation areas • Potential wildli»*  
  > please use the same line spacing for the regular text, otherwise it might mislead a bit, like being titles or something important.

- [ ] **p29** — on: *«used to mask all datasets uniformly. Only pixels corresponding to terrestrial land areas within Sout»*  
  > aah I see .... it is here.   I think the paragrpah needs wording adjustments.

- [ ] **p34** — on: *«variants,»*  
  > which variants? the geo scopes?  if so: keep the wording "geographic scopes"

- [ ] **p34** — on: *«Static Suitability Baseline»*  
  > This needs better wording. The storytelling is a bit confusing.   what is is, (what for), what it does, more details, its results and how it's evaluated.  more or less this structure.

- [ ] **p35**  
  > aaah it comes later...   please rephrase it to make the reading easier

- [ ] **p36** — on: *«Colombia provides a rich and diverse empirical setting in which modelling assumptions and design cho»*  
  > is it not like a repetition of what is written previously?

- [ ] **p38** — on: *«To assess the sensitivity of model performance to the choice of evaluation window, alternative test »*  
  > this paragraph needs to be clearer. It triggers confusion at times. Giving examples might help. Using the same wording as previously might help too.

- [ ] **p42** — on: *«Spatial and temporal consistency. All raster and vector datasets are aligned to a com- mon 1 km refe»*  
  > is it like a repetition of what has been described? where is the limitation so far?

- [ ] **p44** — on: *«interpreted as conditional on past institutional dynamics rather than as unconditional pre- dictions»*  
  > This could kill your goal. Be cautious and decreases the impact using appropriate wording. It sounds like your ML approach is not useful since changes in policy & governance will happen for sure.


## Methodology Questions / Design Justification (29 comments)

*~26 comments asking 'why?', 'how?', 'why 1 km and not 500 m?', etc. These require substantive answers embedded in the text.*

- [ ] **p21** — on: *«supports»*  
  > what does it mean?   is the pixel-level dataset or modeling setting at 1 km? why 1 km and not 2km or not 500m? is there any dataset that needs to be used as it is to decreases interpolation biases as much as possible?

- [ ] **p22** — on: *«and the native resolution of the majority of input datasets. While higher-resolution land- cover pro»*  
  > I think it is still unclear why 1 km.   is the landcover at 1 km?

- [ ] **p22** — on: *«distortions»*  
  > how the distorsion happens? and why at high latitudes?  if you do not want to explain, please cite this information otherwise it sounds like you finds this out here.

- [ ] **p26** — on: *«. P»*  
  > what if a protected area turns into an unprotected area due to whatever reason?   Is this part of the model? if yes, considering the already protected areas would be mandatory, right?   If not part of the model, why?

- [ ] **p28** — on: *«All datasets are harmonised to a common spatial and temporal framework prior to model estimation. Gi»*  
  > I think the spatial grid and what is used for interpolating was introduced already before. But nothing about temporal interpolation / aggregation.   I am still a bit unsure if the section about the reference grid need to be in the data section. It does not describe the data. It gives a new information about how the data will be treated.

- [ ] **p28** — on: *«handled natively by the tree-based machine-learning models employed in this study, which can incorpo»*  
  > this is too early. You are not describing how the models treat the data yet.

- [ ] **p29** — on: *«checked»*  
  > how?

- [ ] **p29** — on: *«Reprojection and resampling»*  
  > you need to explain a bit better the choice of the different interpolation methods. Even though you mention them including the reason. The reasoning behind is not fully clear. a short paragraph of the different interpolation methods and how they might affect the data might be useful. Or you can explain them a bit better in the paragrpahs

- [ ] **p29** — on: *«is constructed.»*  
  > how? using what?

- [ ] **p30** — on: *«within-year reduction.»*  
  > what does it mean? if it was defined before, I must have overlooked it, if so, why not a brief reminder in parenthesis?

- [ ] **p31** — on: *«probability»*  
  > you are mentioned it here as part of the output? how is it defined and built in probability?  is it described in the model section?

- [ ] **p31** — on: *«scales.»*  
  > what kind of scales? temporal? spatial? industrial (small v/s big firms). What is meant by regional dynamics? what is the dynamic? and how does it influence the decisions?

- [ ] **p32** — on: *«that quantify each pixel’s proximity to existing protected areas and infrastructure.»*  
  > is it not like a distance-weight? why is this not also done when smoothing?

- [ ] **p34** — on: *«pansions. Second, small absolute changes in predicted probabilities can correspond to large differen»*  
  > how?

- [ ] **p34** — on: *«predicting non-transition outcomes, reducing their usefulness for identifying future PA ex- pansions»*  
  > how?

- [ ] **p34** — on: *«evaluated»*  
  > the model framework is evaluated? and what is exactly evaluated? and how? I think that mentioning "evaluation" needs to come later. You are first defining/describing the model framework.

- [ ] **p34** — on: *«used»*  
  > why model 1 and model c ?  I understand the C but it looks odd since it does not follow any order or pattern.  why not Model SA for south america? or Model 2 for colombia ?

- [ ] **p35** — on: *«to ensure comparability.»*  
  > if it is the same. why not explaining it here and referring to it here in the other subsection

- [ ] **p35** — on: *«Random Forests provide a flexible, nonparametric benchmark capable of capturing nonlin- earities and»*  
  > how? example?

- [ ] **p37** — on: *«Out-of-fold predictions from temporal cross-validation are retained and used for proba- bility calib»*  
  > how?

- [ ] **p37** — on: *«calibrated using out-of-fold predictions from temporal cross-validation.»*  
  > how?

- [ ] **p38** — on: *«2018–2019), while preserving the same five-year lookahead definition and right-censoring logic. Pred»*  
  > how? I think I am puzzled by the years used. Please clarify

- [ ] **p39** — on: *«areas are computed using a one-year lag, ensuring that the model cannot indirectly infer the locatio»*  
  > why not?

- [ ] **p40** — on: *«This diagnostic verifies that predictive performance is not driven solely by a subset of regions and»*  
  > how?

- [ ] **p40** — on: *«This strategy preserves the true prevalence of protection events while allowing the models to focus »*  
  > how?

- [ ] **p41** — on: *«In addition to discrimination metrics, probability calibration is assessed to ensure interpretabilit»*  
  > how?

- [ ] **p42** — on: *«both Random Forest and LightGBM models. These diagnostics are used to identify the dominant biophysi»*  
  > how are these diagnostic calculated?

- [ ] **p42** — on: *«sensitive»*  
  > how?

- [ ] **p44** — on: *«interpreted as conditional on past institutional dynamics rather than as unconditional pre- dictions»*  
  > This could kill your goal. Be cautious and decreases the impact using appropriate wording. It sounds like your ML approach is not useful since changes in policy & governance will happen for sure.


## Other Comments (96 comments)

*Remaining comments that are more contextual or page-specific.*

- [ ] **p1**  
  > there is a box. Is it just in my laptop?

- [ ] **p21** — on: *«Data and Methodology»*  
  > This is a biiig section. I would suggest splitting into two chapters/sections i.e., Data, Method (or Methodology)

- [ ] **p21** — on: *«3.1.1 Geographic scope and justification»*  
  > You do not need subsections if there is one or 1.5 paragraph.

- [ ] **p21**  
  > It is fine for 1 subsection using 1.5 pages.

- [ ] **p21** — on: *«taining sufficient scale for meaningful spatial generalisation. This design choice avoids con- flati»*  
  > This might be adjusted if you are testing in other areas and getting results that are giving the opposite thinking

- [ ] **p21** — on: *«This grid serves as the common reference framework for integrating heterogeneous geospa- tial datase»*  
  > you might want to introduce the datasets first since the reader may wonder already here about the spatial resolution of the datasets.

- [ ] **p21** — on: *«spatial unit that»*  
  > is the value of the middle of the cell or the average of the cell?

- [ ] **p21** — on: *«repeatedly over time,»*  
  > is it a timeseries in each grid cell?

- [ ] **p22** — on: *«reprojected»*  
  > interpolated?

- [ ] **p22** — on: *«common»*  
  > which one?

- [ ] **p23** — on: *«common»*  
  > which one?

- [ ] **p23** — on: *«All datasets are harmonised to a common spatial framework and temporal structure prior to analysis, »*  
  > any information on temporal coverage and resolution?

- [ ] **p23** — on: *«primary»*  
  > which ones are the secondary ones?

- [ ] **p23** — on: *«3.2.1 Overview of Datasets»*  
  > I would delete the subtitle since it is still about the datasets

- [ ] **p23**  
  > and split it into three tables: for each category

- [ ] **p26** — on: *«risk set,»*  
  > was this term introduced already. If so, ignore my question.

- [ ] **p27**  
  > I think that this section can be together with the previous one as a new chapter since they do not describe the data

- [ ] **p27** — on: *«vegetation indices,»*  
  > what indexes?

- [ ] **p27** — on: *«climate variables,»*  
  > which ones?

- [ ] **p27** — on: *«indicators»*  
  > why indicators?  is it not just if there is o no existing protected areas?

- [ ] **p27** — on: *«human pressure»*  
  > this might mislead. or is it the way it's used usually?

- [ ] **p28** — on: *«Treatment of missing values. Missing values arise primarily from genuine data limita- tions, such as»*  
  > consist of

- [ ] **p28** — on: *«NA»*  
  > NaN ?

- [ ] **p28** — on: *«time-invariant»*  
  > please tell me why long-run climate variables are time invariant? ... what climate variables and what is long run here?

- [ ] **p28** — on: *«historical»*  
  > temporal?

- [ ] **p28** — on: *«Dynamic»*  
  > dynamic means inter annual fluctuations? is that right?

- [ ] **p29** — on: *«in»*  
  > using

- [ ] **p29** — on: *«resampling»*  
  > do you do a new sample?  or you interpolate?

- [ ] **p29** — on: *«1,000 m»*  
  > why in meters here?  you were using km

- [ ] **p29** — on: *«Continuous variables,»*  
  > this is a description of the data and needs to be before.

- [ ] **p29** — on: *«Temporal aggregation»*  
  > be sure that the order of variables (when mentioning them here) is the same as when they were introduced

- [ ] **p30** — on: *«transient artefacts.»*  
  > what are they?

- [ ] **p30** — on: *«reduce sensitivity to outliers and transient artefacts. Wildfire exposure is aggregated cumu- lative»*  
  > for a pixel: maximum would be 365 ?  right?

- [ ] **p30** — on: *«stated in section 3.2.2, there are a total of 413,722,039 pixel–year observations. Transitions into »*  
  > This might need to be somewhere (not here) when introducing what you use and considered in the modelling.

- [ ] **p31** — on: *«set combines (i) baseline cell-level characteristics with (ii) explicitly engineered multi-scale spa»*  
  > you introduce them here but they do not fully match the following subsection. there are three subsections.

- [ ] **p31** — on: *«described in Section 3.2.3.»*  
  > maybe better in parenthesis

- [ ] **p31**  
  > is it really part of the methodology or might it be part as literature review? or intro? it sounds like an standard information rather than something specifically defined for your model framework, right?

- [ ] **p31** — on: *«and wildfire activity. These window sizes correspond to neighbourhood extents of 4 km, 16 km, and 64»*  
  > between 2 pixels, is there a 1 km?   If so:  the distance between the centre and the corner is more than 4, 16 or 64 km.   the projection of the pixels are so that the distance between them can be approximated to a certain value?  like "ca. 1 km"  or "around 1 km" ???

- [ ] **p31** — on: *«Multi-Scale Spatial Smoothing»*  
  > how this smoothing works? is it done three time (4x4, 16x16, and 64x64) ?  or the model chooses one of them?  how is the intertaction of the smoothing outcome with the pointwise value (center)?

- [ ] **p32** — on: *«(dist wdpa)»*  
  > in which program/language/software?

- [ ] **p32** — on: *«This section describes the modelling strategy»*  
  > you need to first introduce the models and then your strategy

- [ ] **p32** — on: *«This section describes the modelling strategy used to predict the emergence of new protected areas. »*  
  > this introductory part needs to be consistent with how you define the subsections. So far, it might be mismatching them or be barely connected with them

- [ ] **p33** — on: *«risk set»*  
  > where is this defined? and why cursive?

- [ ] **p33** — on: *«exclusively on the risk set of unprotected pixels (yi,t = 0), ensuring that predicted probabili- tie»*  
  > how is the probability done?

- [ ] **p33** — on: *«Let»*  
  > ???

- [ ] **p33** — on: *«positive»*  
  > what is positive?

- [ ] **p33** — on: *«In this»*  
  > somewhere, would it be nice to see how your model framework differs from econometric models (the sophisticated ones)

- [ ] **p33** — on: *«baseline hazard functions and covariate effects, the machine-learning approaches employed here allow»*  
  > aha!  your model is a machine learning model. Please define your model before explaining the framework

- [ ] **p33** — on: *«tree-based ensemble models»*  
  > this jumps out of anything.... it sounds odd. Please introduce it more appropriately.

- [ ] **p33** — on: *«functional-form restrictions.»*  
  > which functional-form restrictions?  the assumption? if so: why are they restrictions? in which way?

- [ ] **p34** — on: *«standard classification algorithms»*  
  > which algorithm? and what is standard?

- [ ] **p34** — on: *«rare-event»*  
  > what is a rare event?

- [ ] **p34** — on: *«accuracy-based measures.»*  
  > what is the raw output of the model?

- [ ] **p34** — on: *«To address these issues, the modelling framework focuses on probabilistic outputs and ranking-based »*  
  > please follow the same order you used in the previous paragraph, i.e., you start with class imbalance and then probabilities.

- [ ] **p34** — on: *«Machine-Learning Models»*  
  > before model framework and probably even before

- [ ] **p34** — on: *«under the global 30×30 biodiversity target.»*  
  > I assume this is defined somewhere before Data and Methodology

- [ ] **p34** — on: *«problem»*  
  > is this word commonly used in the computer-sciences field?

- [ ] **p34** — on: *«graphic scopes: (i) a continental-scale model for South America (Model 1), which constitutes the mai»*  
  > keep in mind that this might change afterwards

- [ ] **p34** — on: *«baseline.»*  
  > what is the baseline?

- [ ] **p34** — on: *«Suitability»*  
  > what is suitability here?

- [ ] **p34** — on: *«2000»*  
  > including 2000? even though it was used for training?

- [ ] **p35** — on: *«introduces temporal»*  
  > where? in the baseline model?

- [ ] **p35** — on: *«memory-intensive»*  
  > memory in the model or memory in computing?

- [ ] **p35** — on: *«earities and interactions among predictors. While more memory-intensive than gradient- boosting meth»*  
  > what are they?

- [ ] **p35** — on: *«Random Forest Transition Model»*  
  > same here, structure.  Another thing, I do not see the evaluation you mention in a previous paragraph. Is it an accident?

- [ ] **p35** — on: *«LightGBM»*  
  > where does this acronym come from?

- [ ] **p35** — on: *«LightGBM Transition Model»*  
  > same structure here

- [ ] **p35** — on: *«LightGBM is used due to its computational efficiency and strong performance on large, im- balanced t»*  
  > is this the case with your data, I assume.

- [ ] **p35** — on: *«balanced tabular datasets. The model is trained on the same transition target and temporal regime as»*  
  > there is no definition for training in the random forest subsection

- [ ] **p35** — on: *«gradient-boosted decision trees.»*  
  > what is this

- [ ] **p35** — on: *«scale pos weight»*  
  > where is this?

- [ ] **p35** — on: *«Class imbalance is handled via the scale pos weight parameter, computed as the ratio of negative to »*  
  > i.e., number of negative events divided by positive events?

- [ ] **p35** — on: *«stopping,»*  
  > are other models stopping?

- [ ] **p36** — on: *«estimated»*  
  > models are estimated? or what is estimated?

- [ ] **p36** — on: *«Pipeline»*  
  > you need to introduce this term (commonly used in ML methods) somewhere in the manuscript.

- [ ] **p36** — on: *«velopment under realistic data conditions. This includes experimentation with feature con- struction»*  
  > maybe a bit more of explanation of these terms?

- [ ] **p36** — on: *«expensive»*  
  > how expensive? computationally?  like computing? storage? time used?

- [ ] **p36** — on: *«key drivers relevant»*  
  > which are...?

- [ ] **p36** — on: *«tuning»*  
  > I understand this... but all other readers as well?

- [ ] **p37** — on: *«learning rate, and regularisation parameters, with average precision as the primary optimi- sation c»*  
  > these need to be explained! including numbers

- [ ] **p37** — on: *«sation criterion. The final number of boosting iterations is determined from cross-validation result»*  
  > explain it

- [ ] **p37** — on: *«Final»*  
  > final in a sense of after tunning & adjustments?

- [ ] **p37** — on: *«Platt scaling a»*  
  > explain and/or cite

- [ ] **p38** — on: *«• Training period: 2000–2016 • Early-stopping / validation period: 2014–2016»*  
  > why the training period covers the validation period? are the result not already self contained ? I would be very suprise if the results deviate from the reality (real data)

- [ ] **p38** — on: *«period influence model estimation, tuning, or calibration. This design ensures temporal honesty: per»*  
  > ??? honesty?

- [ ] **p38**  
  > independent?   or another word?

- [ ] **p39** — on: *«temporal leakage.»*  
  > what would be a temporal leakage in your case?

- [ ] **p39** — on: *«is applied unchanged to the test period.»*  
  > the outcome of the calibration or the same calibration process?

- [ ] **p40** — on: *«performance metrics»*  
  > which are?  I have not seen any definition yet about it

- [ ] **p40** — on: *«complementary»*  
  > complementary? what are the principal metrics?

- [ ] **p43** — on: *«rather than biodiversity or other nature-related data per se. Measurement error in predictors can at»*  
  > so therefore....? what next?

- [ ] **p43** — on: *«NA»*  
  > NaN ?  what is expected?  a number?  if so: NaN (not a number)... if not: NA should be OK

- [ ] **p43** — on: *«uine data limitations are retained as NA rather than imputed. Tree-based models naturally accommodat»*  
  > and what do you do with this acknowledge? accept it? or avoiding? or reducing?

- [ ] **p43** — on: *«Non-causal interpretation. The modelling framework is explicitly predictive rather than causal. Esti»*  
  > is it a limitation or a warning for intepretability?

- [ ] **p43**  
  > if limitation... what is it? how it affects, and what to do with it?

- [ ] **p44** — on: *«Reproducibility and Implementation Details»*  
  > are you sure that you want to have a separate dedicated section for this? I have the feeling that this might fit in the part after you describe the model use and before the evaluation?

