import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "6"
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(root_dir)
import json
import math
import time
import random
import argparse
import logging
from datetime import datetime
from functools import partial
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import SimpleITK as sitk
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer,
    Table, TableStyle, KeepInFrame
)

from utils.util import get_rank
from utils.metrics import concordance_index_torch
from lib.model_MOE import FusionModel, ViT_fu, ViT_ct, FusionPipeline

def query_llm_analysis(prompt: str) -> str:
    """
    Optional LLM-based interpretation.
    Requires environment variable: LLM_API_KEY
    """
    api_key = os.getenv("LLM_API_KEY")
    api_url = os.getenv("LLM_API_URL")

    if api_key is None or api_url is None:
        return "LLM interpretation is disabled."

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    payload = {
        "model": "your-llm-model-name",
        "messages": [
            {"role": "system", "content": "You are a medical AI assistant."},
            {"role": "user", "content": prompt}
        ]
    }

    response = requests.post(api_url, headers=headers, json=payload)
    response.raise_for_status()

    content = response.json()["choices"][0]["message"]["content"]
    return json.loads(content).get("analysis", "")

def create_clinical_report(report_data: dict, save_path: str):
    """
    Generate an anonymized AI-assisted clinical report (PDF).
    """

    doc = SimpleDocTemplate(
        save_path,
        pagesize=A4,
        leftMargin=40,
        rightMargin=40,
        topMargin=40,
        bottomMargin=50
    )

    styles = getSampleStyleSheet()
    elements = []

    title_style = ParagraphStyle(
        name="Title",
        parent=styles["Title"],
        alignment=1,
        fontSize=22,
        textColor=colors.HexColor("#005A9C")
    )

    normal_style = ParagraphStyle(
        name="Normal",
        parent=styles["Normal"],
        fontSize=10,
        leading=14
    )

    elements.append(Paragraph("AI Diagnosis Report (Research Use Only)", title_style))
    elements.append(Spacer(1, 30))

    table_data = [[k, str(v)] for k, v in report_data.items()]
    table = Table(table_data, colWidths=[doc.width * 0.35, doc.width * 0.65])
    table.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONT", (0, 0), (-1, -1), "Helvetica", 10),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE")
    ]))

    elements.append(table)
    elements.append(Spacer(1, 40))

    elements.append(Paragraph(
        "Note: This report is generated for research purposes only and "
        "does not constitute clinical advice.",
        normal_style
    ))

    def footer(canvas, doc):
        canvas.setFont("Helvetica-Oblique", 9)
        canvas.drawRightString(
            A4[0] - doc.rightMargin,
            20,
            f"Generated on {datetime.now().strftime('%Y-%m-%d')}"
        )

    doc.build(elements, onFirstPage=footer, onLaterPages=footer)



def create_clinical_report(datas, save_path="clinical_report_final.pdf"):
    """
    Generates a professional clinical diagnosis report in PDF using ReportLab.
    """

    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, KeepInFrame
    )
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    from datetime import datetime
    import json, requests

    # ===============================
    # PDF Setup
    # ===============================
    doc = SimpleDocTemplate(
        save_path,
        pagesize=A4,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=50   # 给 footer 留空间
    )
    elements = []
    styles = getSampleStyleSheet()

    # ===============================
    # Styles
    # ===============================
    title_style = ParagraphStyle(
        name='TitleStyle',
        parent=styles['Title'],
        fontName='Helvetica-Bold',
        fontSize=22,
        alignment=1,
        textColor=colors.HexColor('#005A9C')
    )

    header_style = ParagraphStyle(
        name='HeaderStyle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=16,
        textColor=colors.HexColor('#333333')
    )

    normal_style = ParagraphStyle(
        name='NormalStyle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        alignment=4,
        textColor=colors.HexColor('#333333')
    )

    note_style = ParagraphStyle(
        name='NoteStyle',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=9,
        leading=12,
        alignment=4,
        textColor=colors.HexColor('#333333')
    )

    # ===============================
    # Title
    # ===============================
    elements.append(Paragraph("AI Diagnosis Report", title_style))
    elements.append(Spacer(1, 30))

    # ===============================
    # Patient Info Table
    # ===============================
    table_data = [[k, str(v)] for k, v in datas.items()]

    # 表格宽度 = 正文可用宽度
    total_width = doc.width
    table_col_widths = [total_width * 0.35, total_width * 0.65]

    table = Table(table_data, colWidths=table_col_widths, hAlign='LEFT')
    table.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#DDDDDD'))
    ]))

    elements.append(table)
    elements.append(Spacer(1, 40))

    # ===============================
    # Terminology Notes
    # ===============================
    note_text = (
        "Note:<br/>"
        "• CMS (Consensus Molecular Subtype): A gene expression-based classification for colorectal cancer.<br/>"
        "• MSI (Microsatellite Instability): A marker for genetic instability in tumors. MSI-High status can predict response to immunotherapy. MSS indicates a stable status.<br/>"
        "• DFS/OS (Disease-Free/Overall Survival): Measures of time a patient lives without disease or in total, respectively, after treatment."
    )

    note_para = Paragraph(note_text, note_style)
    elements.append(
        KeepInFrame(
            maxWidth=total_width,
            maxHeight=2000,
            content=[note_para],
            hAlign='LEFT'
        )
    )
    elements.append(Spacer(1, 50))

    # ===============================
    # Interpretation Header
    # ===============================
    elements.append(
        Paragraph(
            "Interpretation of large language model (LLM) Results",
            header_style
        )
    )
    elements.append(Spacer(1, 25))

    # ===============================
    # LLM Response (保持你原逻辑)
    # ===============================
    url = "https://api.modelarts-maas.com/v2/chat/completions"  # API地址
    api_key = ""   #TODO
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}'
    }
    llm_data = { "model": "deepseek-v3.1-terminus",
                "messages": [ {"role": "system", "content": "You are a helpful assistant."}, 
                             {"role": "user", "content": ( "You are a gastroenterology specialist. Based on the CT report below, identify possible misdiagnoses and give a VERY short analysis. "
                                "Return ONLY JSON with the key \\\"analysis\\\". " "The analysis MUST be no longer than 400 words."
                                f"Report: {datas['Age']}-year-old {datas['Sex']}, diagnosed with colorectal cancer. Based on the CT images, the preliminary assessment indicates: " 
                                f"{datas['T stage']} stage, {datas['N stage']} stage, {datas['M stage']} stage, with an overall TNM stage of {datas['TNM stage']}. " 
                                f"CMS subtype {datas['CMS']}, MMR showing {datas['MSI']} status, {datas['Disease-Free Survival (DFS)']} DFS and {datas['Overall survival (OS)']} OS prognosis." )} ], 
                "thinking": {"type": "enabled"} }

    response = requests.post(url, headers=headers, data=json.dumps(llm_data), verify=False)
    response_json = response.json()
    content_str = response_json["choices"][0]["message"]["content"]
    content_dict = json.loads(content_str)
    interpretation = content_dict["analysis"]
    # print("LLM raw output:\n", raw_content)
    elements.append(
        KeepInFrame(
            maxWidth=total_width,
            maxHeight=4000,
            content=[Paragraph(interpretation, normal_style)],
            hAlign='LEFT'
        )
    )

    # ===============================
    # Footer (固定右下角)
    # ===============================
    def draw_footer(canvas, doc):
        footer_text = f"Report Date: {datetime.now().strftime('%Y-%m-%d')}"
        canvas.setFont("Helvetica-Oblique", 9)
        canvas.setFillColor(colors.HexColor('#333333'))
        canvas.drawRightString(
            A4[0] - doc.rightMargin,
            20,
            footer_text
        )

    # ===============================
    # Build PDF
    # ===============================
    doc.build(
        elements,
        onFirstPage=draw_footer,
        onLaterPages=draw_footer
    )


@torch.no_grad()
def evaluate(model_a,model_b,fusion_model, task_name, data_path, args,flag=0):
    model_a.eval()
    model_b.eval()
    fusion_model.eval()
    with torch.no_grad():
        img = sitk.ReadImage(data_path)
        img = torch.from_numpy(sitk.GetArrayFromImage(img).astype(np.float32))
        img_tensor = torch.unsqueeze(torch.unsqueeze(img,0),0).to(torch.float32).cuda()
        pred = fusion_model(img_tensor)
    map_list = json.load(open('utils/map_list.json','r'))
    if task_name in ['dfs', 'os']:
        threshold = map_list[task_name]
        if pred>threshold:
            true_label = 'high risk'
        else:
            true_label = 'low risk'
        pred_prob = 0
    else:
        probs = F.softmax(pred, dim=-1)
        pred_class = probs.argmax(dim=-1).item()
        pred_prob = probs.max().item()
        true_label = map_list[task_name][str(pred_class)]
    
    return {'true_label':true_label, 'prob':round(pred_prob,2)}

def main(args):
    # device = torch.device(args.device)
    seed = args.seed + get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    batch_size = args.batch_size
    summary_writer = SummaryWriter(args.log_dir)
    model_a = ViT_ct(
    image_size = 256,          # image size
    frames = 32,               # number of frames
    image_patch_size = 16,     # image patch size
    frame_patch_size = 16,      # frame patch size
    dim = 1024,
    depth = 24,
    heads = 16,
    emb_dropout = 0.1,
    norm_layer=partial(torch.nn.LayerNorm, eps=1e-6)) ### 196
    model_a = model_a.cuda()

    model_b = ViT_fu(
    image_size = 256,          # image size
    frames = 32,               # number of frames
    image_patch_size = 16,     # image patch size
    frame_patch_size = 16,      # frame patch size
    dim = 1024,
    depth = 24,
    heads = 16,
    norm_layer=partial(torch.nn.LayerNorm, eps=1e-6)) ### 196
    model_b = model_b.cuda()
    
    pred_results ={}

    for i in range(len(args.tasks)):
        task_name = args.tasks[i]
        num_class = args.num_classes[i]
        pred_results[task_name] = {}
        ct_weights_path = os.path.join(args.pretrained_root, task_name.split('.')[-1] + '.pth')
        fusion_model = FusionModel(input_dim_a=1024, input_dim_b=1024, classes=num_class)
        pipeline = FusionPipeline(model_a, model_b, fusion_model,num_class)
        model_dict = pipeline.state_dict()
        if os.path.exists(ct_weights_path):
            weights_dict = torch.load(ct_weights_path, map_location='cpu')
            pipeline.load_state_dict(weights_dict)
            print('load sucessed!!!!')
        pipeline = pipeline.cuda()
        pred_results[task_name] = evaluate(model_a,model_b,pipeline, task_name, data_path=args.img_root, args=args,flag=1)
    print(pred_results)
    ### create_clinical_report
    datas = {
        "Patient ID": args.Patient_ID,
        "Name": args.Patient_name,
        "Age": args.Patient_age,
        "Sex": args.Patient_sex,
        "T stage": pred_results['tnm.t']['true_label'] + " (Probability:" + str(pred_results['tnm.t']['prob']) + ')',
        "N stage": pred_results['tnm.n']['true_label'] + " (Probability:" + str(pred_results['tnm.n']['prob']) + ')',
        "M stage": pred_results['tnm.m']['true_label'] + " (Probability:" + str(pred_results['tnm.m']['prob']) + ')',
        "TNM stage": pred_results['tnm.tnm']['true_label'] + " (Probability:" +str( pred_results['tnm.tnm']['prob']) + ')',
        "CMS": pred_results['cms']['true_label'] + " (Probability:" + str(pred_results['cms']['prob']) + ')',
        "MSI": pred_results['msi']['true_label'] + " (Probability:" + str(pred_results['msi']['prob']) + ')',
        "Disease-Free Survival (DFS)": pred_results['dfs']['true_label'],
        "Overall survival (OS)": pred_results['os']['true_label']
    }
    create_clinical_report(datas=datas, save_path=f"{args.Patient_ID}_{args.Patient_name}_{args.Patient_age}.pdf")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str,
                        default='mae',
                        help='model name:[resnet18_fe,resnet34,resnet50_fe,resnet101,resnext50,resnext50_fe,resnext152_fe,resnet18_fe,resnet34_fe]')
    parser.add_argument('--vit', default=True, type=bool)
    parser.add_argument('--pretrained_root', default=f'./weights', type=str)
    parser.add_argument('--num_classes', type=int, default=[2,2,2,4,2,4,1,1])
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--scale', type=list, default=[1])
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--shape', type=tuple, default=(32,256,256))
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--lrf', type=float, default=0.1)
    parser.add_argument('--alpha', type=float, default=0.4)
    parser.add_argument('--mask_ratio', default=0., type=float, help='mask ratio of pretrain')
    parser.add_argument('--img_root', type=str, required=True)
    parser.add_argument('--Patient_ID', type=str, default="")
    parser.add_argument('--Patient_name', type=str, default="")
    parser.add_argument('--Patient_age', type=str, default="")
    parser.add_argument('--Patient_sex', type=str, default="")
    parser.add_argument('--tasks', type=list, default=["tnm.t", "tnm.n", "tnm.m", "tnm.tnm", "msi", "cms", "dfs", "os"])
    parser.add_argument('--log_path', type=str,
                        default='./logs',
                        help='path to log')
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:4', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    exp_name = opt.model_name+'-'+str(opt.lr)
    opt.log_dir = opt.log_path +'/'+ exp_name + '/logs'
    opt.model_dir = opt.log_path +'/'+ exp_name + '/models'
    os.makedirs(opt.log_dir, exist_ok=True)
    os.makedirs(opt.model_dir, exist_ok=True)

    logging.basicConfig(filename=os.path.join(opt.log_dir, 'train_log.log'),
                        format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                        level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')
    logging.info('Hyperparameter setting{}'.format(opt))
    main(opt)