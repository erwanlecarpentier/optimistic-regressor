# Optimistic Regressor

## Principle 

Notations:

Empirical distribution:
<img src="https://latex.codecogs.com/svg.latex?\Large&space;q" title="\Large x=\frac{-b\pm\sqrt{b^2-4ac}}{2a}" /><br/>
Target:
<img src="https://latex.codecogs.com/svg.latex?\Large&space;f = y_{max} - y" title="\Large x=\frac{-b\pm\sqrt{b^2-4ac}}{2a}" />


,e.g.,
<img src="https://latex.codecogs.com/svg.latex?\Large&space;f = \frac{1}{1-\gamma} - \left(r + \gamma \max_{a'} Q(s', a')\right)" title="\Large x=\frac{-b\pm\sqrt{b^2-4ac}}{2a}" /><br/>



Optimistic loss:

<img src="https://latex.codecogs.com/svg.latex?\Large&space;L(\theta)=\|f - f_{\theta}(x)\|_{2, q}^2 +" title="\Large x=\frac{-b\pm\sqrt{b^2-4ac}}{2a}" /><br/>

## Using venv python library

For venv documentation: [click here](https://docs.python.org/3/library/venv.html)  
Activating:

	source ./venv/bin/activate
	
Deactivating:

	deactivate

## Dependencies

| Package | Version |
| ----------- | ----------- |
| Pillow | 8.0.1 |
| certifi | 2020.6.20 |
| configparser | 5.0.1 |
| cycler | 0.10.0 |
| dataclasses | 0.6 |
| future | 0.18.2 |
| kiwisolver | 1.3.1 |
| matplotlib | 3.3.2 |
| numpy | 1.19.4 |
| pip | 20.2.4 |
| pyparsing | 2.4.7 |
| python-dateutil | 2.8.1 |
| setuptools | 50.3.2 |
| six | 1.15.0 |
| torch | 1.7.0 |
| torchvision | 0.8.1 |
| typing-extensions | 3.7.4.3 |

