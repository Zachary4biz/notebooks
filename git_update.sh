
# also add an alias in ~/.bashrc
# ==> alias gup="sh git_update.sh" 
####################################
# use ssh to access the repository.
# git remote rm origin
# git remote add origin git@github.com:Zachary4biz/notebooks.git
#

commit_info=$1
if [ ! -n "$commit_info" ]; then
    echo "use default commit_info"
    commit_info="default"
fi
 
git add .
git commit -am "$commit_info"
git push

