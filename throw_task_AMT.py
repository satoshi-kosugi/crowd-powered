import xml.etree.ElementTree as ET
import boto3
import time
import datetime
import numpy
import pickle
import cv2
import os
import random
import math
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

def throw_task_AMT(scp_image_names, image_names, label_hitid_record, label_hitid_record_name, AMT_config_original, default_values, interps, prev_i, predicted_images):
    AMT_config = AMT_config_original.copy()
    validate_idxs = list(range(5))
    validate_image_names = []
    validate_default_values = []
    simlink_dir = os.path.basename(label_hitid_record_name)
    if not os.path.exists("html_images/"+simlink_dir):
        os.mkdir("html_images/"+simlink_dir)

    now_ = scp_image_names[0].split("/")[1].split("_")[1] + "_" \
                + scp_image_names[0].split("/")[1].split("_")[2]

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
    image_names_correspondence, choices_correspondence, judgements_record = make_simlink(simlink_dir, scp_image_names, image_names, prev_i, predicted_images)
    QuestionHtml = create_question_html(scp_image_names, image_names, validate_image_names, default_values, validate_default_values, simlink_dir)
    QuestionHtmlDebug = create_question_html_debug(scp_image_names, image_names, validate_image_names, default_values, validate_default_values, simlink_dir, prev_i, image_names_correspondence, choices_correspondence, judgements_record)

    approved_workers = []

    while True:
        client = get_client(AMT_config["sandbox"])
        print('AvailableBalance:', client.get_account_balance()['AvailableBalance'])

        if unfinished_hitid is None:
            if not os.path.exists("html/" + os.path.basename(label_hitid_record_name)):
                os.makedirs("html/" + os.path.basename(label_hitid_record_name))
            with open("html/" + os.path.basename(label_hitid_record_name) + '/' + now_ + "_" + str(repeat_task) + ".html", 'w') as f:
                f.write(QuestionHtml)
            with open("html/" + os.path.basename(label_hitid_record_name) + '/' + now_ + "_" + str(repeat_task) + "_debug.html", 'w') as f:
                f.write(QuestionHtmlDebug)

            res = client.create_hit(
                Title=AMT_config["Title"],
                Description=AMT_config["Description"],
                Keywords=AMT_config["Keywords"],
                Reward=str(float(AMT_config["Reward"])*len(image_names)),
                MaxAssignments=AMT_config["MaxAssignments"],
                LifetimeInSeconds=AMT_config["LifetimeInSeconds"],
                AssignmentDurationInSeconds=AMT_config["AssignmentDurationInSeconds"],
                AutoApprovalDelayInSeconds=AMT_config["AutoApprovalDelayInSeconds"],
                Question=QuestionHtml
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
        results_raw = {}

        for assignment in res2["Assignments"]:
            results[assignment["WorkerId"]] = {}
            results_raw[assignment["WorkerId"]] = {}
            results_raw[assignment["WorkerId"]]["ReasonForRejection"] = []
            same_choices_check = []
            user_validated = True
            reflect_check = {}
            for answer in ET.fromstring(assignment["Answer"]):
                if "reflectCheck" in answer[0].text:
                    reflect_check[answer[0].text[-1]] = answer[1].text
                    results_raw[assignment["WorkerId"]][answer[0].text] = answer[1].text
            for answer in ET.fromstring(assignment["Answer"]):
                if answer[0].text == "deviceCheck":
                    results[assignment["WorkerId"]][answer[0].text] = answer[1].text
                    results_raw[assignment["WorkerId"]][answer[0].text] = answer[1].text
                    if answer[1].text == "sp":
                        user_validated = False
                        results_raw[assignment["WorkerId"]]["ReasonForRejection"].append("sp")
                elif not "reflectCheck" in answer[0].text:
                    if reflect_check[answer[0].text] == "original":
                        results[assignment["WorkerId"]][image_names_correspondence[answer[0].text]] = choices_correspondence[answer[0].text][answer[1].text]
                    else:
                        results[assignment["WorkerId"]][image_names_correspondence[answer[0].text]] = choices_correspondence[answer[0].text][str(5-int(answer[1].text))]
                    results_raw[assignment["WorkerId"]][answer[0].text] = answer[1].text
                    if image_names_correspondence[answer[0].text][:3] != "val":
                        same_choices_check.append(int(answer[1].text))
            for image_name in results[assignment["WorkerId"]].keys():
                if image_name[:3] == "val":
                    if results[assignment["WorkerId"]][image_name] != 2:
                        user_validated = False
                        results_raw[assignment["WorkerId"]]["ReasonForRejection"].append("check task")

            if len(set(same_choices_check)) == 1:
                user_validated = False
                results_raw[assignment["WorkerId"]]["ReasonForRejection"].append("same choices")
            if len(results[assignment["WorkerId"]].keys()) != len(image_names)+2:
                user_validated = False
                results_raw[assignment["WorkerId"]]["ReasonForRejection"].append("incomplete answer")
            if assignment["WorkerId"] in approved_workers:
                user_validated = False
                results_raw[assignment["WorkerId"]]["ReasonForRejection"].append("duplicate answer")

            if user_validated:
                for image_name in image_names:
                    if len(prev_i.keys()) == 0:
                        value = interps[image_name][results[assignment["WorkerId"]][image_name]] * 31
                        validated_results[image_name].append(value)
                    else:
                        value = results[assignment["WorkerId"]][image_name]
                        validated_results[image_name].append(value)

            hit_assignment_id = unfinished_hitid + "_" + assignment["AssignmentId"] + "_" + assignment["WorkerId"]
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
        print("Results_raw:", results_raw)
        with open("log/"+unfinished_hitid+".pickle", 'wb') as f:
            pickle.dump(results, f)
        with open("log/"+unfinished_hitid+"_raw.pickle", 'wb') as f:
            pickle.dump(results_raw, f)

        label_hitid_record["HITId"][now_][unfinished_hitid] = "finished"
        with open(label_hitid_record_name+".pickle", 'wb') as f:
            pickle.dump(label_hitid_record, f)
        unfinished_hitid = None

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
<h3 style="margin-bottom:40px;">If the parameters are adjusted randomly, the reward may not be paid.</h3>
<form name='mturk_form' method='post' id='mturk_form' action='https://www.mturk.com/mturk/externalSubmit'>
"""
# <form name='mturk_form' method='post' id='mturk_form' action='https://www.mturk.com/mturk/externalSubmit' style="background-color:#929292">


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
  label, input[type='checkbox'] {
    cursor: pointer;
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
      <input type="range" id="{2}" name="{2}" min="0" max="5" step="1" value="{5}"></input>
    </div>
  <div class="col-xs-3"></div>
</div>
<div class="col-xs-2" style="display: none;">
  <input type="radio" name="reflectCheck{1}" id="original{1}" value="original"/>
  <input type="radio" name="reflectCheck{1}" id="reflect{1}" value="reflect"/>
</div>
"""

each_question_item_debug = \
"""
  <div class="col-xs-2" style="padding: 0;">
    <div class="col-xs-12" style="padding: 1px;">
        <img style="max-height:350px" src="{11}/{2}_{3}_{5}_{6}.jpg"></img>
    </div>
    {8}:{9}({10})
  </div>
"""

each_question_start = \
"""
<div class="col-xs-12" style="padding:0;">
    <div class="col-xs-12"><h2 style="margin-top:15px;">Q{1}</h2></div>
"""

each_question_end = \
"""
</div>
"""

script_start = """
    <div class="col-xs-2" style="display: none;">
      sp<input type="radio" name="deviceCheck" id="deviceSp" value="sp"/>
      pc<input type="radio" name="deviceCheck" id="devicePc" value="pc"/>
    </div>
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
<script>
  if (window.matchMedia && window.matchMedia('(max-device-width: 640px)').matches) {
    document.getElementById("deviceSp").checked = true;
  } else {
    document.getElementById("devicePc").checked = true;
  }
</script>
"""
onload_start = """
window.onload = () => {{
"""
onload_end = """
}}
"""

each_question_script = \
"""
  const inputElem{1} = document.getElementById('{1}');
  const random{1} = Math.random();

  if (random{1} >= 0.5){{
    document.getElementById("original{1}").checked = true;
  }} else{{
    document.getElementById("reflect{1}").checked = true;
  }}

  const setCurrentValue{1} = (val{1}) => {{
      if (random{1} >= 0.5){{
          document.getElementById("{0}").src="{5}/{4}/{1}_{2}_{3}_"+val{1};
      }} else{{
          document.getElementById("{0}").src="{5}/{4}/{1}_{2}_{3}_"+(5-val{1});
      }}
  }}

  const rangeOnChange{1} = (e{1}) =>{{
      if (random{1} >= 0.5){{
          document.getElementById("{0}").src="{5}/{4}/{1}_{2}_{3}_"+e{1}.target.value;
      }} else{{
          document.getElementById("{0}").src="{5}/{4}/{1}_{2}_{3}_"+(5-e{1}.target.value);
      }}
  }}
"""

each_question_onload = """
inputElem{1}.addEventListener('input', rangeOnChange{1});
setCurrentValue{1}(inputElem{1}.value);
"""

def create_question_html(scp_image_names, image_names, validate_image_names, default_values, validate_default_values, simlink_dir):
    repeat = scp_image_names[0].split("/")[1].split("_")[1]
    step = scp_image_names[0].split("/")[1].split("_")[2]

    html_txt = header
    for i in range(len(image_names)+1):
        html_txt += each_question.format("image"+str(i+1),i+1,i+1,repeat,step,0)

    html_txt += script_start

    for i in range(len(image_names)+1):
        html_txt += each_question_script.format("image"+str(i+1), i+1, repeat, step, simlink_dir, fileServer_config["httpURL"])

    html_txt += onload_start

    for i in range(len(image_names)+1):
        html_txt += each_question_onload.format("image"+str(i+1), i+1)

    html_txt += onload_end
    html_txt += script_end
    html_txt += footer

    return html_txt

def create_question_html_debug(scp_image_names, image_names, validate_image_names, default_values, validate_default_values, simlink_dir, prev_i, image_names_correspondence, choices_correspondence, judgements_record):
    repeat = scp_image_names[0].split("/")[1].split("_")[1]
    step = scp_image_names[0].split("/")[1].split("_")[2]
    image_names_correspondence_swap = {v: k for k, v in image_names_correspondence.items()}

    html_txt = header
    for i in range(len(image_names)):
        html_txt += each_question_start.format("___",image_names[i],image_names[i],repeat,0,step)
        splits = sorted(list(range(32)) + [prev_i[image_names[i]] * 31])
        for i_, j in enumerate(splits):
            keys = []
            for k, v in choices_correspondence[image_names_correspondence_swap[image_names[i]]].items():
                if v == j:
                    keys.append(k)
            keys = sorted(keys)
            if j != prev_i[image_names[i]] * 31:
                html_txt += each_question_item_debug.format("___",image_names[i],image_names[i],repeat,0,step,int(j),simlink_dir,int(j),"_".join(keys),", ".join(judgements_record[image_names[i]][j]),fileServer_config["httpURL"])
            else:
                html_txt += each_question_item_debug.format("___",image_names[i],image_names[i],repeat,0,step,-1,simlink_dir,math.floor(prev_i[image_names[i]]*31*100)/100.,"_".join(keys),"",fileServer_config["httpURL"])
            if i_ % 6 == 5:
                html_txt += """<div class="col-xs-12" style="padding: 10px;"></div>"""
        html_txt += each_question_end

    html_txt += each_question_end

    html_txt += script_start
    html_txt += onload_start
    html_txt += onload_end
    html_txt += script_end
    html_txt += footer

    return html_txt


def spearman(imagex, imagey, patch_size=8):
    imagex_tile = imagex.reshape((1, -1))
    imagey_tile = imagey.reshape((1, -1))
    return numpy.corrcoef(imagex_tile, imagey_tile)[0,1]


def judge_image(image1, changed_area, original_image, previous_image, illumination_, LIME):
    illumination = illumination_ / 255.
    illumination[illumination>=0.5] = 1
    illumination[illumination<0.5] = 0
    illumination_edge = numpy.zeros_like(illumination)
    illumination_edge[1:] += numpy.abs(illumination[1:] - illumination[:-1])
    illumination_edge[:,1:] += numpy.abs(illumination[:,1:] - illumination[:,:-1])
    illumination_edge = (numpy.clip(illumination_edge, 0, 1)*255).astype(numpy.uint8)
    illumination_edge = cv2.dilate(illumination_edge, numpy.ones((15,15),numpy.uint8))[:,:,0]
    illumination_edge[illumination[:,:,0]==0] = 0
    luminance = image1[:,:,0] * 0.114 + image1[:,:,1] * 0.587 + image1[:,:,2] * 0.299

    judgements = []

    if ((image1.min(axis=2) > 245) * (LIME.min(axis=2) < 245)).mean() >= 0.05:
        judgements.append("shirotobi LIME")
    if ((image1.max(axis=2) < 75) * (original_image.max(axis=2) > 75)).mean() >= 0.1:
        judgements.append("kuroochi original_image")
    if (luminance > 245).mean() >= 0.4:
        judgements.append("shirotobi luminance")
    if ((image1.max(axis=2) < 30) * changed_area).sum() >= 1/2:
        judgements.append("kuroochi changed_area")
    if ((image1.min(axis=2) > 245) * changed_area).sum() >= 1/2:
        judgements.append("shirotobi changed_area")
    if ((image1.max(axis=2) < 75) * (original_image.max(axis=2) > 75)).mean() >= 0.1:
        judgements.append("kuroochi original_image")
    if ((image1.max(axis=2) < 75) * (previous_image.max(axis=2) > 75)).mean() >= 0.1:
        judgements.append("kuroochi previous_image")
    if ((image1.min(axis=2) > 245) * (original_image.min(axis=2) < 245) * (changed_area / changed_area.max())).mean() >= 0.2:
        judgements.append("shirotobi original_image")
    if ((cv2.cvtColor(image1.astype(numpy.uint8), cv2.COLOR_BGR2HSV)[:,:,1] < 30) * (cv2.cvtColor(original_image.astype(numpy.uint8), cv2.COLOR_BGR2HSV)[:,:,1] > 30) * (image1.min(axis=2) > 200) * (changed_area / changed_area.max())).mean() >= 0.2:
        judgements.append("shirotobi original_saturation")
    if numpy.corrcoef(original_image.max(axis=2).flatten(), image1.max(axis=2).flatten())[0,1] <= 0:
        judgements.append("corrcoef")
    if numpy.corrcoef(original_image.max(axis=2)[illumination_edge==255], image1.max(axis=2)[illumination_edge==255])[0,1] <= 0.1:
        judgements.append("local corrcoef")
    if numpy.corrcoef(original_image.max(axis=2)[previous_image.max(axis=2) > 75], image1.max(axis=2)[previous_image.max(axis=2) > 75])[0,1] <= 0.2:
        judgements.append("corrcoef w/o kuroochi")
    return judgements

def make_simlink(simlink_dir, scp_image_names, image_names_, prev_i, predicted_images):
    repeat = scp_image_names[0].split("/")[1].split("_")[1]
    step = scp_image_names[0].split("/")[1].split("_")[2]
    splits = [0, 6, 12, 18, 24, 31]

    validation_i = 3
    image_names = image_names_[:validation_i] + ["val"] + image_names_[validation_i:]

    image_names_correspondence = {}
    choices_correspondence = {}
    judgements_record = {}

    for i, image_name in enumerate(image_names):
        if image_name in prev_i.keys():
            splits = []
            maes = [0]
            judgementss = []
            image0 = predicted_images["html_images/{0}_{1}_{2}_{3}.jpg".format(image_name, repeat, step, 0)] * 1.
            for i_ in range(32):
                image1 = predicted_images["html_images/{0}_{1}_{2}_{3}.jpg".format(image_name, repeat, step, i_)] * 1.
                maes.append(maes[-1] + ((image1 - image0) ** 2).sum() ** 0.5)
                image0 = image1.copy()
            maes = numpy.array(maes[1:])

            image_left = predicted_images["html_images/{0}_{1}_{2}_{3}.jpg".format(image_name, repeat, step, 0)] * 1.
            image_right = predicted_images["html_images/{0}_{1}_{2}_{3}.jpg".format(image_name, repeat, step, 31)] * 1.
            changed_area = ((image_right * 1. - image_left) ** 2).sum(axis=2) ** 0.5
            changed_area = changed_area / changed_area.sum()
            original_image = predicted_images["html_images/{0}_{1}_{2}_{3}.jpg".format(image_name, 1, 0, -1)] * 1.
            previous_image = predicted_images["html_images/{0}_{1}_{2}_{3}.jpg".format(image_name, repeat, step, -1)] * 1.
            illumination = cv2.imread("LIME/illumination/"+image_name+".png")
            LIME = cv2.imread("LIME/results/"+image_name+".png")

            left_mae = maes[0]
            left_i = -1
            for i_ in range(32):
                image1 = predicted_images["html_images/{0}_{1}_{2}_{3}.jpg".format(image_name, repeat, step, i_)] * 1.
                judgements = judge_image(image1, changed_area, original_image, previous_image, illumination, LIME)
                judgementss.append(judgements)
            for i_ in range(32):
                if len(judgementss[i_]) == 0:
                    left_mae = maes[i_]
                    left_i = i_
                    break
            judgements_record[image_name] = judgementss
            right_mae = maes[-1]
            right_i = -1
            for i_ in range(31, -1, -1):
                image1 = predicted_images["html_images/{0}_{1}_{2}_{3}.jpg".format(image_name, repeat, step, i_)] * 1.
                if len(judgementss[i_]) == 0:
                    right_mae = maes[i_]
                    right_i = i_
                    break

            diff_now_prev_min = 1000000000000000
            diff_now_prev_argmin = -1
            image_prev = predicted_images["html_images/{0}_{1}_{2}_{3}.jpg".format(image_name, repeat, step, -1)] * 1.
            for i_ in range(5):
                splits.append(((numpy.arange(32) - (left_i + (right_i - left_i) * i_ / 4.))**2).argmin())
            if left_i == -1 and right_i == -1:
                splits = [-1 for i_ in range(5)]
            splits = sorted(list(set(splits)))
            if len(splits) == 6:
                if abs(prev_i[image_name] * 31 - splits[0]) > abs(prev_i[image_name] * 31 - splits[-1]):
                    splits = splits[1:]
                else:
                    splits = splits[:5]

            if len(splits) >= 3:
                if prev_i[image_name] * 31 <= splits[0]:
                    splits = [-1] * (6-len(splits)) + splits
                elif prev_i[image_name] * 31 >= splits[-1]:
                    splits = splits + [-1] * (6-len(splits))
                else:
                    for i_ in range(5):
                        if prev_i[image_name] * 31 >= splits[i_] and prev_i[image_name] * 31 <= splits[i_+1]:
                            splits = splits[:i_+1] + [-1] * (6-len(splits)) + splits[i_+1:]
                            break
            elif len(splits) == 2:
                if prev_i[image_name] * 31 <= splits[0]:
                    splits = [-1] * 2 + [splits[0]] * 2 + [splits[1]] * 2
                elif prev_i[image_name] * 31 >= splits[-1]:
                    splits = [splits[0]] * 2 + [splits[1]] * 2 + [-1] * 2
                else:
                    splits = [splits[0]] * 2 + [-1] * 2 + [splits[1]] * 2
            else:
                splits = splits * 3 + [-1] * 3

        image_names_correspondence[str(i+1)] = image_name
        choices_correspondence[str(i+1)] = {}
        if image_name[:3] == "val":
            random_change = random.randint(1, 4)
            random_filp = random.randint(0, 1)
            for j in range(len(splits)):
                if random_filp == 0:
                    if j <= random_change-2:
                        os.symlink("../{0}_{1}.jpg".format(image_name, 0), os.path.join("html_images/", simlink_dir, "{0}_{1}_{2}_{3}".format(i+1, repeat, step, j)))
                        choices_correspondence[str(i+1)][str(j)] = 0
                    elif j == random_change-1:
                        os.symlink("../{0}_{1}.jpg".format(image_name, 1), os.path.join("html_images/", simlink_dir, "{0}_{1}_{2}_{3}".format(i+1, repeat, step, j)))
                        choices_correspondence[str(i+1)][str(j)] = 1
                    elif j == random_change:
                        os.symlink("../{0}_{1}.jpg".format(image_name, 2), os.path.join("html_images/", simlink_dir, "{0}_{1}_{2}_{3}".format(i+1, repeat, step, j)))
                        choices_correspondence[str(i+1)][str(j)] = 2
                    elif j == random_change+1:
                        os.symlink("../{0}_{1}.jpg".format(image_name, 3), os.path.join("html_images/", simlink_dir, "{0}_{1}_{2}_{3}".format(i+1, repeat, step, j)))
                        choices_correspondence[str(i+1)][str(j)] = 3
                    else:
                        os.symlink("../{0}_{1}.jpg".format(image_name, 4), os.path.join("html_images/", simlink_dir, "{0}_{1}_{2}_{3}".format(i+1, repeat, step, j)))
                        choices_correspondence[str(i+1)][str(j)] = 4
                else:
                    if j <= random_change-2:
                        os.symlink("../{0}_{1}.jpg".format(image_name, 4), os.path.join("html_images/", simlink_dir, "{0}_{1}_{2}_{3}".format(i+1, repeat, step, j)))
                        choices_correspondence[str(i+1)][str(j)] = 4
                    elif j == random_change-1:
                        os.symlink("../{0}_{1}.jpg".format(image_name, 3), os.path.join("html_images/", simlink_dir, "{0}_{1}_{2}_{3}".format(i+1, repeat, step, j)))
                        choices_correspondence[str(i+1)][str(j)] = 3
                    elif j == random_change:
                        os.symlink("../{0}_{1}.jpg".format(image_name, 2), os.path.join("html_images/", simlink_dir, "{0}_{1}_{2}_{3}".format(i+1, repeat, step, j)))
                        choices_correspondence[str(i+1)][str(j)] = 2
                    elif j == random_change+1:
                        os.symlink("../{0}_{1}.jpg".format(image_name, 1), os.path.join("html_images/", simlink_dir, "{0}_{1}_{2}_{3}".format(i+1, repeat, step, j)))
                        choices_correspondence[str(i+1)][str(j)] = 1
                    else:
                        os.symlink("../{0}_{1}.jpg".format(image_name, 0), os.path.join("html_images/", simlink_dir, "{0}_{1}_{2}_{3}".format(i+1, repeat, step, j)))
                        choices_correspondence[str(i+1)][str(j)] = 0
        else:
            random_filp = random.randint(0, 1)
            if random_filp == 0:
                for j, split in enumerate(splits):
                    os.symlink("../{0}_{1}_{2}_{3}.jpg".format(image_name, repeat, step, split), os.path.join("html_images/", simlink_dir, "{0}_{1}_{2}_{3}".format(i+1, repeat, step, j)))
                    if split == -1:
                        choices_correspondence[str(i+1)][str(j)] = prev_i[image_name] * 31
                    else:
                        choices_correspondence[str(i+1)][str(j)] = split
            else:
                for j, split in enumerate(list(reversed(splits))):
                    os.symlink("../{0}_{1}_{2}_{3}.jpg".format(image_name, repeat, step, split), os.path.join("html_images/", simlink_dir, "{0}_{1}_{2}_{3}".format(i+1, repeat, step, j)))
                    if split == -1:
                        choices_correspondence[str(i+1)][str(j)] = prev_i[image_name] * 31
                    else:
                        choices_correspondence[str(i+1)][str(j)] = split

    os.system((" rsync -rl -e \"ssh -p {0}\" html_images/" + simlink_dir + " {1}@{2}:{3}").format(fileServer_config["sshPort"], fileServer_config["sshUsername"], fileServer_config["sshIP"], fileServer_config["sshDirectory"]))
    return image_names_correspondence, choices_correspondence, judgements_record


def unblock_workers(workers, AMT_config):
    client = get_client(AMT_config["sandbox"])

    for worker in workers:
        client.delete_worker_block(WorkerId=worker, Reason="I have lifted the temporary block. I would be very grateful if you could help me again!")
