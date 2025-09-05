# MonoCLUE: Object-aware Clustering Enhances Monocular 3D Object Detection

## Introduction
This repository provides the official implementation of [MonoCLUE: Object-aware Clustering Enhances Monocular 3D Object Detection](https://arxiv.org) based on the excellent work [MonoDGP](https://github.com/PuFanqi23/MonoDGP). In this work, we propose a DETR-based monocular 3D detection framework that strengthens visual reasoning by leveraging clustering and scene memory, enabling robust performance under occlusion and limited visibility.
<div align="center"> <img src="figures/overall_architecture.png" width="600" height="auto"/> </div> 

## Main Figures
<div align="center"> <img src="figures/explanation.png" width="600" height="auto"/> </div>

## Main Result
The official results :
<table>
    <tr>
        <td rowspan="2",div align="center">Models</td>
        <td colspan="3",div align="center">Val, AP<sub>3D|R40</sub></td>   
        <td rowspan="2",div align="center">Logs</td>
        <td rowspan="2",div align="center">Ckpts</td>
    </tr>
    <tr>
        <td div align="center">Easy</td> 
        <td div align="center">Mod.</td> 
        <td div align="center">Hard</td> 
    </tr>
    <tr>
        <td rowspan="4",div align="center">MonoCLUE</td>
        <td div align="center">33.7426%</td> 
        <td div align="center">24.1090%</td> 
        <td div align="center">20.5883%</td> 
        <td div align="center"><a href="https://drive.google.com/file/d/1ccwmKmxjJMtiD5GAYMlB9Acz_sV2gtwJ/view?usp=sharing">log</a></td>
        <td div align="center"><a href="https://drive.google.com/file/d/1Nddzx3xDE0DPZzVluR9HEYRgH2wALU9z/view?usp=sharing">ckpt</a></td>
    </tr>  
  <tr>
        <td div align="center">31.5802%</td> 
        <td div align="center">23.5648%</td> 
        <td div align="center">20.2746%</td> 
        <td div align="center"><a href="https://drive.google.com/file/d/1mjk457aBjxs6a3Lf-biX10_YzhW2th_U/view?usp=sharing">log</a></td>
        <td div align="center"><a href="https://drive.google.com/file/d/1eCON928oVFTL2U64qZotWYhRCRopldxY/view?usp=sharing">ckpt</a></td>
    </tr>  
</table>

"MonoCLUE":

<table>
    <tr>
        <td rowspan="2" align="center"><b>Models</b></td>
        <td colspan="3" align="center"><b>Val, AP<sub>3D&#124;R40</sub></b></td>   
    </tr>
    <tr>
        <td align="center"><b>Easy</b></td> 
        <td align="center"><b>Mod.</b></td> 
        <td align="center"><b>Hard</b></td> 
    </tr>
    <tr>
        <td align="center">Ours</td>
        <td align="center">33.7426% (+2.98%)</td> 
        <td align="center">24.1090% (+1.76%)</td> 
        <td align="center">20.5883% (+1.57%)</td> 
    </tr>  
</table>

Model pth: 
<div>
  <td div align="center"><a href="https://drive.google.com/file/d/183zRv7EaR3ReS4QA9KTfLPbRRxwjHcqU/view?usp=drive_link">Model_pth</a></td>
</div>

Log: 
<div>
  <td div align="center"><a href="https://drive.google.com/file/d/17rU7GzQx1XdSXQqmWOot9efmh7fgEf88/view?usp=drive_link">Train log</a></td>
</div>
