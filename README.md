# Reimplementation of Evaluating Progress in Automatic Chest X-Ray Radiology Report Generation


<div style="color: black; border: 1px solid red; padding: 10px; background-color: #f8d7da;">
  <strong>WARNING: Work in Progress</strong>
  <p> The code currently works (see <a href="https://github.com/SummitStudiosDev/RadCliQ/blob/main/test.ipynb">test.ipynb</a>), but may change in the future. <br> I intend to make it into a pypi package eventually. Requirements have not been listed, please look at the requirements of the dependencies for now. (Specifically the radgraph package's dependencies)</p>
</div>
<br>


This repository reimplements the RadCliQ composite metric as defined in the paper "[Evaluating Progress in Automatic Chest 
X-Ray Radiology Report Generation](https://www.medrxiv.org/content/10.1101/2022.08.30.22279318v1)".

This reimplementation allows for the calculation of RadCliQ via a function all in memory, while the original needed the ground truth and predicted reports to be saved to csvs in order to calculate RadCliQ. The original also extracted embeddings from Chexbert and then saved them into a file, while this reimplementation keeps them in memory.

Contains code for computing the individual metric scores that are used to compute RadCliQ. The individual metrics are:
* BLEU
* BERTscore
* CheXbert labeler vector similarity
* RadGraph entity and relation F1

Much of the code is liberally borrowed from [rajpurkarlab/CXR-Report-Metric](https://github.com/rajpurkarlab/CXR-Report-Metric/).

The code for computing the CheXbert metric score is adapted from
[stanfordmlgroup/CheXbert](https://github.com/stanfordmlgroup/CheXbert).
Chexbert weights are not included in this repo. Download the
CheXbert model checkpoint [here](https://stanfordmedicine.box.com/s/c3stck6w6dol3h36grdc97xoydzxd7w9) and drag them into src/radcliq/CXR-Report-Metric (may be changed later)

## Changes
Replaced the [original](https://physionet.org/content/radgraph/1.0.0/) RadGraph weights with the [RadGraph-XL](https://aclanthology.org/2024.findings-acl.765/) weights. Code to calculate RadGraph is from [Stanford-AIMI/radgraph](https://github.com/Stanford-AIMI/radgraph). This has the advantage of being more up to date and no longer requiring credentialed access, but scores may vary slightly from the original. 


<a name="license"></a>

# License
This repository is made publicly available under the MIT License.

<a name="citing"></a>

# Citing
If you are using this repo, please cite the original paper:
```
@article {Yu2022.08.30.22279318,
	author = {Yu, Feiyang and Endo, Mark and Krishnan, Rayan and Pan, Ian and Tsai, Andy and Reis, Eduardo Pontes and Fonseca, Eduardo Kaiser Ururahy Nunes and Ho Lee, Henrique Min and Abad, Zahra Shakeri Hossein and Ng, Andrew Y. and Langlotz, Curtis P. and Venugopal, Vasantha Kumar and Rajpurkar, Pranav},
	title = {Evaluating Progress in Automatic Chest X-Ray Radiology Report Generation},
	elocation-id = {2022.08.30.22279318},
	year = {2022},
	doi = {10.1101/2022.08.30.22279318},
	publisher = {Cold Spring Harbor Laboratory Press},
	URL = {https://www.medrxiv.org/content/early/2022/08/31/2022.08.30.22279318},
	eprint = {https://www.medrxiv.org/content/early/2022/08/31/2022.08.30.22279318.full.pdf},
	journal = {medRxiv}
}
```

Additionally, if you found this implementation useful, please mention:
```
@misc{SummitStudiosDev/RadCliQ, author = {Cody Chen}, title = {Remplementation of [Evaluating Progress in Automatic Chest X-Ray Radiology Report Generation
]}, year = {2025}, url = {https://github.com/SummitStudiosDev/RadCliQ} }
```