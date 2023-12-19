# Crowd-Powered Photo Enhancement Featuring an Active Learning Based Local Filter
This is the official implementation of the [paper](https://ieeexplore.ieee.org/abstract/document/10005188) in TCSVT2023.

## Requirements
- Python 3.5.9
- MATLAB R2016b

To use MATLAB functions from Python, at the MATLAB command prompt,
```Shell
cd (fullfile(matlabroot,'extern','engines','python'))
system('python setup.py install')
```
To install the Python libraries,
```Shell
pip install --upgrade pip
pip install -r requirements.txt
pip install ./sequential-line-search_my
pip install megengine -f https://megengine.org.cn/whl/mge.html
```

## Experiments using BIQME
1. Different number of pixels L `--num_pixels`.
```Shell
python evaluation/RunExperiment.py --num_pixels 4 --mode BIQME
```

2. Without active learning `--woactivelearning` or the illumination map `--woilluminationmap`.
```Shell
python evaluation/RunExperiment.py --mode BIQME --woactivelearning
python evaluation/RunExperiment.py --mode BIQME --woilluminationmap
```

3. Previous local filters. `--filter_type` can be set to `global`, `graduated`, `elliptical`, `cubic10`, and `cubic20`.
```Shell
python evaluation/RunExperimentLPF.py --mode BIQME --filter_type graduated
```
The BIQME scores are saved as .npy files, and the file name is printed at the end of the process. By setting the file names in `BIQME_scores` of `draw_graph.py` and executing `draw_graph.py`, the graph of the BIQME scores is output.

## Experiments on Amazon Mechanical Turk
To conduct experiments on Amazon Mechanical Turk (AMT), `config.py` needs to be set first. For AMT's API, please create an Amazon Web Service account, get a pair of an access key ID and a secret access key, and set them in `"aws_access_key_id"` and `"aws_secret_access_key"` of `AMTAPI_config`.

To publish images to crowd workers, a server with HTTP access is needed. Please set the server's IP address, port number, user name, and URL in `"sshIP"`, `"sshPort"`, `"sshUsername"`, and `"httpURL"` of `fileServer_config`. `"sshDirectory"` should be set to the directory pointed by `"httpURL"`.

By setting `AMT_config["sandbox"]` as `True`, you can check the interface for the crowd workers in the [sandbox environment](https://workersandbox.mturk.com/) without paying the fee.

1. Our method.
```Shell
python evaluation/RunExperiment.py --mode AMT --image_names 0,1,2,3,4
```
2. Sequential Line Search.
```Shell
python evaluation/RunExperimentSLS.py --mode AMT --image_names 0,1,2,3,4
```

## Demo
You can adjust a sequence of single sliders by yourself.
```Shell
python evaluation/RunExperiment.py --mode self --image_names 19
```

## Results
Results of our method and compared previous methods can be downloaded from [here](https://www.hal.t.u-tokyo.ac.jp/~kosugi/crowd-powered/results.zip).

## Citation
If you find our research useful in your research, please consider citing:

    @article{kosugi2023crowd,
        title={Crowd-Powered Photo Enhancement Featuring an Active Learning Based Local Filter},
        author={Kosugi, Satoshi and Yamasaki, Toshihiko},
        journal={IEEE Transactions on Circuits and Systems for Video Technology},
        volume={33},
        number={7},
        pages={6493--6501},
        year={2023}
    }
