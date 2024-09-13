## Step 1: Setting up
1. Create a new folder and navigate to it in your terminal
2. Initialize git using this command: `git init`
3. Add this repository to your remote: `git remote add origin https://github.com/La-PleIAde/La-PleIAde.git`
4. Load the project: `git pull origin main`  (this is for the `main` branch, you might want to load another branch, feel free)
5. Congratulations, now you can open the project in your IDE and do the stuff!

* Note: if you use an advanced IDE such as VSCode or PyCharm, there are ways to load the project directly from remote. So, if you can do how to do so, you can skip this step


## Step 2: Contributing
0. Before changing anything make shure your project is up-to-date: `git pull`
1. Create a separate branch and checkout: `git checkout -b your_branch_name` (your branch name should represent the feature you're working on, but of course, be short)
2. Do your changes. Do not forget to test befote committing :)
3. If you have created or added any files that are important to be in the project, add them to git using `git add your_file` or `git add --all` if you want to add everything. 
	*Warning:* do not add or commit sensible data.
4. Commit your changes with meaningfull message: `git commit -m'your very important commit message'`
5. Ready? Push: `git push origin your_branch_name` (to upload changes to the remote)
6. In GitHub, create a new pull request and assign me as a reviewer
7. Well done? Maybe... if not yet, you'll have further instructions. And... after all your changes will be mergen into the main branch :P


### Face any problems? Have any qiestions? Feel free to ask!!!
