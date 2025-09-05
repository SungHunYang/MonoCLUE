# MonoCLUE

This repository provides the official implementation of [MonoCLUE: Object-aware Clustering Enhances Monocular 3D Object Detection](https://arxiv.org) based on the excellent work [MonoDGP](https://github.com/PuFanqi23/MonoDGP). In this work, we propose a DETR-based monocular 3D detection framework that strengthens visual reasoning by leveraging clustering and scene memory, enabling robust performance under occlusion and limited visibility.

## Results
"MonoDGP" in the paper:

<table>
    <tr>
        <td rowspan="2",div align="center">Models</td>
        <td colspan="3",div align="center">Val, AP<sub>3D|R40</sub></td>   
    </tr>
    <tr>
        <td div align="center">Easy</td> 
        <td div align="center">Mod.</td> 
        <td div align="center">Hard</td> 
    </tr>
    <tr>
        <td rowspan="4",div align="center">MonoDGP</td>
        <td div align="center">30.7624%</td> 
        <td div align="center">22.3421%</td> 
        <td div align="center">19.0144%</td> 
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
