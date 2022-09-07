my_bluemix_apikey=g8oY01PfCf4ZSl74CngcoC7v2hqQx9NQhlrrn4h8-eXd
my_bluemix_user='ayon01051998@gmail.com'
my_blumix_password='Mello@15'
my_instance_crn='crn:v1:bluemix:public:sql-query:us-south:a/bdcd440fdcdf403792aa931788c0e43b:7dda6373-0d9b-4b65-acd5-c685cc8c3320::'
my_target_cos_endpoint='s3.ap.cloud-object-storage.appdomain.cloud'
my_target_cos_bucket='enginx2019'
my_target_cos_prefix=''
my_target_cos="cos://${my_target_cos_endpoint}/${my_target_cos_bucket}/${my_target_cos_prefix}"

#get IAM aut token for api key
my_access_token1=`curl --request POST http://iam.ng.bluemix.net/oidc/token \
-H "Content-Type: application/x-www-form-urlencoded; charset=utf-8" \
-d "grant_type=urn:ibm:params:oauth:grant-type:apikey" \
-d "response_type=cloud_iam" \
-d "apikey=${my_bluemix_key}" \
-d "bx:bx" \
| jq -r '.access_token'`
my_access_token1="Bearer $my_access_token1"

curl "https://iam.cloud.ibm.com/identity/token" \
        -d "apikey=g8oY01PfCf4ZSl74CngcoC7v2hqQx9NQhlrrn4h8-eXd&grant_type=urn%3Aibm%3Aparams%3Aoauth%3Agrant-type%3Aapikey" \
        -H "Content-Type: application/x-www-form-urlencoded" \
        -H "Authorization: Basic Yng6Yng="
