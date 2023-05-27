# Instructions

Note: This repository uses Git LSF for the knowledge base of the agent. Please 
keep that in mind when enabling the option. https://git-lfs.com/

To execute this code first all dependencies must be installed:

```
pip install -r requirements.txt
```

Also the Cython code must be compiled, to do this run:

```
python setup.py build_ext --inplace
```

Finally the code can be executed with:

```
python main.py
```

And to check the plots of the obtained data:

```
python plotter.py
```