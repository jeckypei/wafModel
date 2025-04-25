#get data set from github
##    https://github.com/openappsec/waf-comparison-project 

if [ ! -e data/Legitimate ];then 
cd ./data/ 

wget https://downloads.openappsec.io/waf-comparison-project/legitimate.zip
wget https://downloads.openappsec.io/waf-comparison-project/malicious.zip

unzip legitimate.zip
unzip malicious.zip
fi
