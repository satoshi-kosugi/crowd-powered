import xml.etree.ElementTree as ET
import boto3
import time
import datetime
import numpy
import pickle
import cv2
from config import AMTAPI_config, fileServer_config

def get_client(sandbox_flag):
    if sandbox_flag:
        return boto3.client("mturk",
                            aws_access_key_id=AMTAPI_config["aws_access_key_id"],
                            aws_secret_access_key=AMTAPI_config["aws_secret_access_key"],
                            region_name="us-east-1",
                            endpoint_url="https://mturk-requester-sandbox.us-east-1.amazonaws.com"
        )
    else:
        return boto3.client("mturk",
                            aws_access_key_id=AMTAPI_config["aws_access_key_id"],
                            aws_secret_access_key=AMTAPI_config["aws_secret_access_key"],
                            region_name="us-east-1",
                            endpoint_url="https://mturk-requester.us-east-1.amazonaws.com"
        )

def throw_task_AMT(scp_image_names, image_names, label_hitid_record, label_hitid_record_name, AMT_config_original, default_values):
    AMT_config = AMT_config_original.copy()
    validate_idxs = list(range(len(image_names)))
    validate_image_names = []
    validate_default_values = []
    for validate_idx in validate_idxs:
        validate_image_names.append(image_names[validate_idx])
        validate_default_values.append(0)

    now_ = scp_image_names[0].split("/")[1].split("_")[1]

    unfinished_hitid = None
    if now_ in label_hitid_record["HITId"].keys():
        for hitid in label_hitid_record["HITId"][now_].keys():
            if label_hitid_record["HITId"][now_][hitid] == "unfinished":
                unfinished_hitid = hitid
    else:
        label_hitid_record["HITId"][now_] = {}

    validated_results = {}
    for image_name in image_names:
        validated_results[image_name] = []

    repeat_task = 0
    while True:
        client = get_client(AMT_config["sandbox"])
        print('AvailableBalance:', client.get_account_balance()['AvailableBalance'])

        if unfinished_hitid is None:
            res = client.create_hit(
                Title=AMT_config["Title"],
                Description=AMT_config["Description"],
                Keywords=AMT_config["Keywords"],
                Reward=str(float(AMT_config["Reward"])*len(image_names)),
                MaxAssignments=AMT_config["MaxAssignments"],
                LifetimeInSeconds=AMT_config["LifetimeInSeconds"],
                AssignmentDurationInSeconds=AMT_config["AssignmentDurationInSeconds"],
                AutoApprovalDelayInSeconds=AMT_config["AutoApprovalDelayInSeconds"],
                Question=create_question_html(scp_image_names, image_names, validate_image_names, default_values, validate_default_values),
            )
            unfinished_hitid = res["HIT"]["HITId"]
            print("Status Code:", res["ResponseMetadata"]["HTTPStatusCode"])

        print("MaxAssignments:", AMT_config["MaxAssignments"])
        print("HIT ID:", unfinished_hitid)
        label_hitid_record["HITId"][now_][unfinished_hitid] = "unfinished"
        with open(label_hitid_record_name+".pickle", 'wb') as f:
            pickle.dump(label_hitid_record, f)

        while True:
            time.sleep(60)
            res2 = client.list_assignments_for_hit(HITId=unfinished_hitid)
            print(datetime.datetime.now(), len(res2["Assignments"]), res2["ResponseMetadata"]["HTTPStatusCode"])
            if len(res2["Assignments"]) >= AMT_config["MaxAssignments"]:
                break

        results = {}

        for assignment in res2["Assignments"]:
            user_validated = False
            results[assignment["WorkerId"]] = {}
            for answer in ET.fromstring(assignment["Answer"]):
                results[assignment["WorkerId"]][answer[0].text] = int(answer[1].text)

            for image_name in image_names:
                value = results[assignment["WorkerId"]][image_name]
                valvalue = 31 - results[assignment["WorkerId"]][image_name+"val"]
                if abs(value * 1. - valvalue) <= 8:
                    validated_results[image_name].append((value+valvalue)/2.)
                    user_validated = True

            hit_assignment_id = unfinished_hitid + "_" + assignment["AssignmentId"]
            client.approve_assignment(AssignmentId=assignment["AssignmentId"])
            if not hit_assignment_id in label_hitid_record["approve_reject"]:
                label_hitid_record["approve_reject"][hit_assignment_id] = "approve"
            if not user_validated:
                block_message = "Thanks for carrying out my task!\
                    You completed my task without any problem and helped me so much!\
                    I have blocked you now, but this is a temporary block.\
                    I have published other tasks, and to reduce bias by workers,\
                    I limit the number of times each worker can carry out my task.\
                    I will unblock you after the other tasks are completed (probably in a few hours),\
                    so I would be very grateful if you could help me again after being unblocked."
                client.create_worker_block(WorkerId=assignment["WorkerId"], Reason=block_message)
                label_hitid_record["blocked_workers"].append(assignment["WorkerId"])

        print("Results:", results)
        with open("log/"+unfinished_hitid+".pickle", 'wb') as f:
            pickle.dump(results, f)

        label_hitid_record["HITId"][now_][unfinished_hitid] = "finished"
        with open(label_hitid_record_name+".pickle", 'wb') as f:
            pickle.dump(label_hitid_record, f)
        unfinished_hitid = None

        validated = True
        print("# validated answers:")
        print(len(validated_results[image_names[0]]))
        if len(validated_results[image_names[0]]) == AMT_config_original["MaxAssignments"]:
            validated = True
        else:
            validated = False
            AMT_config["MaxAssignments"] = AMT_config_original["MaxAssignments"] - len(validated_results[image_names[0]])

        if validated:
            break
        else:
            print("Throw the same task again.")
            repeat_task += 1

    print("validated results:", validated_results)

    return_results = {}
    for image_name in image_names:
        return_results[image_name] = numpy.median(numpy.array(validated_results[image_name]))

    print("median:", return_results)

    return return_results


header = \
"""
<HTMLQuestion xmlns="http://mechanicalturk.amazonaws.com/AWSMechanicalTurkDataSchemas/2011-11-11/HTMLQuestion.xsd">
    <HTMLContent><![CDATA[
<!DOCTYPE html>
<html>
<head>
    <meta http-equiv='Content-Type' content='text/html; charset=UTF-8'/>
    <script type='text/javascript' src='https://s3.amazonaws.com/mturk-public/externalHIT_v1.js'></script>
</head>
<body>
<main>
<link crossorigin="anonymous" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.0.3/css/bootstrap.min.css" integrity="sha384-IS73LIqjtYesmURkDE9MXKbXqYA8rvKEp/ghicjem7Vc3mGRdQRptJSz60tvrB6+" rel="stylesheet" /><!-- The following snippet enables the 'responsive' behavior on smaller screens -->
<meta content="width=device-width,initial-scale=1" name="viewport" /><!-- Instructions -->
<section class="container" id="TaggingOfAnImage">
<div class="row">
<h1>Please adjust the photo retouching parameters for the best results.</h1>
<h3>If the parameters are adjusted randomly, the reward may not be paid.</h3>
<form name='mturk_form' method='post' id='mturk_form' action='https://www.mturk.com/mturk/externalSubmit'>
<div class="col-xs-12" style="margin-top:50px;"></div>
"""


footer = \
"""

</section>
</main>
<p><a> <!-- End Image Tagging Layout --><!-- Open internal style sheet -->
<style type="text/css">#collapseTrigger{
  color:#fff;
  display: block;
  text-decoration: none;
}
#submitButton{
  white-space: normal;
}
.image{
  margin-bottom: 15px;
}
.radio:first-of-type{
  margin-top: -5px;
}
</style>
<!-- Close internal style sheet --><!-- Please note that Bootstrap CSS/JS and JQuery are 3rd party libraries that may update their url/code at any time. Amazon Mechanical Turk (MTurk) is including these libraries as a default option for you, but is not responsible for any changes to the external libraries -->
<script src="https://code.jquery.com/jquery-3.1.0.min.js" integrity="sha256-cCueBR6CsyA4/9szpPfrX3s49M9vUU5BgtiJj06wt/s=" crossorigin="anonymous"></script><script src="https://maxcdn.bootstrapcdn.com/bootstrap/3.0.3/js/bootstrap.min.js" integrity="sha384-s1ITto93iSMDxlp/79qhWHi+LsIi9Gx6yL+cOKDuymvihkfol83TYbLbOw+W/wv4" crossorigin="anonymous"></script><script>
  $(document).ready(function() {
    // Instructions expand/collapse
    var content = $('#instructionBody');
    var trigger = $('#collapseTrigger');
    content.hide();
    $('.collapse-text').text('(Click to expand)');
    trigger.click(function(){
      content.toggle();
      var isVisible = content.is(':visible');
      if(isVisible){
        $('.collapse-text').text('(Click to collapse)');
      }else{
        $('.collapse-text').text('(Click to expand)');
      }
    });
    // end expand/collapse
  });
</script>
</a></p>
</body>

<style type="text/css">
  input[type="range"] {
    -webkit-appearance: none;
    appearance: none;
    cursor: pointer;
    outline: none;
    height: 10px;
    width: 100%;
    background: #8acdff;
    border-radius: 10px;
    border: solid 3px #dff1ff;
  }
  input[type="range"]::-webkit-slider-thumb {
    -webkit-appearance: none;
    background: #53aeff;
    width: 24px;
    height: 24px;
    border-radius: 50%;
    box-shadow: 0px 3px 6px 0px rgba(0, 0, 0, 0.15);
  }
  input[type="range"]::-moz-range-thumb {
    background: #53aeff;
    width: 24px;
    height: 24px;
    border-radius: 50%;
    box-shadow: 0px 3px 6px 0px rgba(0, 0, 0, 0.15);
    border: none;
  }
  input[type="range"]::-moz-focus-outer {
    border: 0;
  }
  input[type="range"]:active::-webkit-slider-thumb {
    box-shadow: 0px 5px 10px -2px rgba(0, 0, 0, 0.3);
  }
  img {
    max-width: 100%;
    height: auto;
    width /***/:auto;
  }
</style>
</html>
]]>
    </HTMLContent>
    <FrameHeight>800</FrameHeight>
</HTMLQuestion>
"""

each_question = \
"""
<div class="col-xs-12">
  <div class="col-xs-3"></div>
  <div class="col-xs-6 fields" style="text-align:center">
    <img id="{0}" style="max-height:350px"></img>
  </div>
  <div class="col-xs-4"></div>
  </div>
<div class="col-xs-12" style="padding-bottom: 30px;">
  <div class="col-xs-3"></div>
    <div class="col-xs-1"><h2 style="margin-top:5px;">Q{1}</h2></div>
    <div class="col-xs-5 fields" style="margin-top:15px;">
      <input type="range" id="{2}" name="{2}" min="0" max="31" step="1" value="{4}"></input>
    </div>
  <div class="col-xs-3"></div>
</div>
"""

validate_each_question = \
"""
<div class="col-xs-12">
  <div class="col-xs-3"></div>
  <div class="col-xs-6 fields" style="text-align:center">
    <img id="{0}val" style="max-height:350px"></img>
  </div>
  <div class="col-xs-4"></div>
  </div>
<div class="col-xs-12" style="padding-bottom: 30px;">
  <div class="col-xs-3"></div>
    <div class="col-xs-1"><h2 style="margin-top:5px;">Q{1}</h2></div>
    <div class="col-xs-5 fields" style="margin-top:15px;">
      <input type="range" id="{2}val" name="{2}val" min="0" max="31" step="1" value="{4}"></input>
    </div>
  <div class="col-xs-3"></div>
</div>
"""

each_question_script = \
"""
  const inputElem{2} = document.getElementById('{2}');

  const setCurrentValue{2} = (val{2}) => {{
  document.getElementById("{0}").src="{5}/{2}_{3}_"+val{2}+".jpg";
  }}

  const rangeOnChange{2} = (e{2}) =>{{
  document.getElementById("{0}").src="{5}/{2}_{3}_"+e{2}.target.value+".jpg";
  }}
"""

validate_each_question_script = \
"""
  const inputElem{2}val = document.getElementById('{2}val');

  const setCurrentValue{2}val = (val{2}val) => {{
  document.getElementById("{0}val").src="{5}/{2}_{3}_"+(31-val{2}val)+".jpg";
  }}

  const rangeOnChange{2}val = (e{2}val) =>{{
  document.getElementById("{0}val").src="{5}/{2}_{3}_"+(31-e{2}val.target.value)+".jpg";
  }}
"""

each_question_onload = """
inputElem{2}.addEventListener('input', rangeOnChange{2});
setCurrentValue{2}(inputElem{2}.value);
"""

validate_each_question_onload = """
inputElem{2}val.addEventListener('input', rangeOnChange{2}val);
setCurrentValue{2}val(inputElem{2}val.value);
"""

script_start = """
    <center>
        <form name='mturk_form' method='post' id='mturk_form' action='https://www.mturk.com/mturk/externalSubmit'>
            <p><input type='submit' id='submitButton' value='Submit' /></p>
            <input type='hidden' value='' name='assignmentId' id='assignmentId'/>
        </form>
    </center>
</form>
</div>
<script language='Javascript'>turkSetAssignmentID();</script>
<script>
"""
script_end = """
</script>
"""
onload_start = """
window.onload = () => {{
"""
onload_end = """
}}
"""

def create_question_html(scp_image_names, image_names, validate_image_names, default_values, validate_default_values):
    repeat = scp_image_names[0].split("/")[1].split("_")[1]

    html_txt = header
    for i, image_name in enumerate(image_names):
        html_txt += each_question.format("image"+image_name,i+1,image_name,repeat,default_values[i])
    for i, image_name in enumerate(validate_image_names):
        html_txt += validate_each_question.format("image"+image_name,i+1+len(image_names),image_name,repeat,validate_default_values[i])

    html_txt += script_start

    for i, image_name in enumerate(image_names):
        html_txt += each_question_script.format("image"+image_name,i+1,image_name,repeat,default_values[i],fileServer_config["httpURL"])
    for i, image_name in enumerate(validate_image_names):
        html_txt += validate_each_question_script.format("image"+image_name,i+1+len(image_names),image_name,repeat,validate_default_values[i],fileServer_config["httpURL"])

    html_txt += onload_start

    for i, image_name in enumerate(image_names):
        html_txt += each_question_onload.format("image"+image_name,i+1,image_name,repeat,default_values[i])
    for i, image_name in enumerate(validate_image_names):
        html_txt += validate_each_question_onload.format("image"+image_name,i+1+len(image_names),image_name,repeat,validate_default_values[i])


    html_txt += onload_end
    html_txt += script_end
    html_txt += footer

    return html_txt

def unblock_workers(workers, AMT_config):
    client = get_client(AMT_config["sandbox"])

    for worker in workers:
        client.delete_worker_block(WorkerId=worker, Reason="I have lifted the temporary block. I would be very grateful if you could help me again!")
