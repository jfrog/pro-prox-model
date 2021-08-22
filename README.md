## How to set up your project and connect to Valohai
Contact IT and create a dedicated github repo for your project.
Use the following command in terminal:
```
git clone git@github.com:jfrog/ds-template.git <YOUR-REPO-NAME>
```
Enter the directory you've just created using:
```
cd <YOUR-REPO-NAME>
```
In order to start the process of connecting your newly pulled template to your dedicated repo for your project, Run the following command:
```
bash run.sh
```
Follow the step in the run.sh process that asks you to enter your dedicated repo name.
> ![Screen Shot 2021-08-16 at 10 03 13](https://user-images.githubusercontent.com/46316863/129524187-c8e9af4b-baf7-4b22-ad47-42a7c8ddf1ce.png)
 
You now have your project repository all set up with the basic files needed to start your project and connect to Valohai.
Go to the Valohai UI and create a corresponding Valohai project:
> ![Screen Shot 2021-08-16 at 10 07 49](https://user-images.githubusercontent.com/46316863/129524673-46a909a7-37cb-4627-b1ad-87ba449fe94a.png)

Choose a project name (its easier to keep the same name for your repo/valohai project) and save the newly created project.
Go back to your terminal, exit your project folder with the following command:
```
cd ..
```
Use the following command to create an SSH private and public keys, to connect your git repo to your new Valohai project:
```
ssh-keygen -t rsa -b 4096 -N '' -f your-project-name-deploy-key

```
You should see something like this:
> ![Screen Shot 2021-08-16 at 10 32 57](https://user-images.githubusercontent.com/46316863/129527442-4f598944-e868-4f97-b674-fec2166c2933.png)

<b>Make sure you don't save those keys in your git repo!</b>
Send the <b>public key (the file that ends with '.pub')</b> to one of our IT guys/gals and ask them to attach it to your git-repo.
After that, Go back to Valohai. In your project, go into the Settings tab and after that, click the Repository sub-tab. There, you should attach your git-repo link:
> ![Screen Shot 2021-08-16 at 10 22 22](https://user-images.githubusercontent.com/46316863/129527630-e30cd686-5cd2-465a-b4c7-ef94e4d6627c.png)
* Under "URL", attach your git-repo SSH link (e.g. "git+ssh://git@github.com/jfrog/csat-model.git").
* Under "Fetch references", state the branches you would like to get pulled into your valohai project (e.g. "dev,staging,prod").
* Under "SSH private key", attach the content of the private key you've just generated. to get this content, use the following command in the terminal:
```
cat your-project-name-deploy-key
```
Save the settings, and click the "fetch repository" button on the top right to pull the project content into Valohai.
Open the local project via your preferrable IDE, and use the project preferences/settings in order to set up your virtual-environment.
While the VENV is activated, use the following command in your project root:
```
pip install -r requirements.txt
```
If at any point you'd like to add new libraries/packages, simply install them and add them to the requirements file (Don't forget to push your changes and fetch them from Valohai as well!)
<br/>
At this point, you'd want to set up your environment variables. You should find your local .env file in your project folder,
Populate it as you would with any other project. If you have a .env file ready from another project, you can simply copy and paste it.
<br/>
In order for that variables to be set within Valohai, go to Settings -> Environment Variables:
> ![Screen Shot 2021-08-17 at 11 17 07](https://user-images.githubusercontent.com/46316863/129690065-120193e0-bb81-46cd-9583-ec0739d5357e.png)

