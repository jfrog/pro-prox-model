#!/bin/sh

git remote rm origin
rm -rf .git
echo Initializing .env file...
touch .env
echo Creating a Data folder...
mkdir data
echo Please enter your git repo name, in case you dont have any, please ask the IT to create one for this project
read project_name
git=".git"
project_name_full="$project_name$git"
echo $project_name_full
echo Initializing git repo for the project $project_name2
git init
git add .
git commit -m "my first commit"
git remote add origin git@github.com:jfrog/$project_name_full
git push --set-upstream origin master