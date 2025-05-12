from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from collections import defaultdict, Counter
from itertools import combinations, permutations

import cv2
import numpy as numpy
import os
import math

app = FastAPI()

app.add_middleware(
  CORSMiddleware,
  allow_origins=["*"],
  allow_credentials=True,
  allow_methods=["*"],
  allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.post("/help-core/")
async def help(file: list[UploadFile] = File(...), job: str = Form(...)):
  try:
    displays = []
    for file in files:
      contents = await file.read()
      npimg = np.frombuffer(contents, np.uint8)
      display = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
      displays.append(display)

    base_path = "static"
    table_template = cv2.imread(os.path.join(base_path, "templates", "table.png"))
    empty_template = cv2.imread(os.path.join(base_path, "templates", "empty.png"))
    job_dir = os.path.join(base_path, "jobs", job)

    results = []
    for display in displays:
      table = get_table(display, table_template)
      cores = get_cores(table, empty_template)
      cores = get_enhanced_cores(cores)
      core_skills = get_skills(cores)
      parsed_skills = parse_skills(core_skills, job_dir)

      _, skill_names = zip(*parsed_skills)
      target_skills = ["레조네이트", "데스 블로섬"]
      combinations = find_combinations(skill_names, target_skills)

      results.append(combinations)

    return JSONResponse(content={
      "success": True,
      "message": "코어 조합 분석 완료",
      "results": results
    })

  except Exception as e:
    return JSONResponse(status_code=500, content={
      "success": False,
      "message": f"분석 중 오류 발생: {str(e)}"
    })

def get_table(display, template):
  gray_display = cv2.cvtColor(display, cv2.COLOR_BGR2GRAY)
  gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
  result = cv2.matchTemplate(gray_display, gray_template, cv2.TM_CCOEFF_NORMED)
  _, _, _, max_loc = cv2.minMaxLoc(result)
  top_left = max_loc
  h, w = gray_template.shape
  return display[top_left[1]:top_left[1]+h, top_left[0]:top_left[0]+w]

def get_cores(table, template):
  cores = []
  gray_table = cv2.cvtColor(table, cv2.COLOR_BGR2GRAY)
  gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
  blurred = cv2.GaussianBlur(table, (5, 5), 0)
  edges = cv2.Canny(blurred, 90, 180)
  contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  for cnt in contours:
    if cv2.contourArea(cnt) < 500:
      continue
    approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
    if 5 <= len(approx) <= 9:
      x, y, w, h = cv2.boundingRect(approx)
      if 0.7 < (w / h) < 1.3:
        cropped = gray_table[y:y+h, x:x+w]
        resized_template = cv2.resize(gray_template, (w, h))
        result = cv2.matchTemplate(cropped, resized_template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)
        if max_val < 0.5:
          cores.append(table[y:y+h, x:x+w])
  return cores

def get_enhanced_cores(cores):
  enhanced_cores = []
  dark_color = np.array([0, 0, 0])
  light_color = np.array([94, 115, 113])
  for c in cores:
    mask = cv2.inRange(c, dark_color, light_color)
    h, w = mask.shape
    x = w // 6
    y = (h // 2) - 5
    white, black = 0, 0
    for i in range(10):
      if mask[y+i, x] == 255:
        white += 1
      else:
        black += 1
    if white > black:
      enhanced_cores.append(c)
  return enhanced_cores

def get_skills(cores):
  skills = []
  for c in cores:
    hsv = cv2.cvtColor(c, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([0, 0, 0]))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
      continue
    biggest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(biggest)
    skills.append(c[y:y+h, x:x+w])
  return skills

def parse_skills(core_skills, job_folder):
  def apply_border(image):
    color = [221, 221, 204]
    white = [255, 255, 255]
    for x in range(1, 31):
      image[1, x] = color
      image[30, x] = color
      image[2, x] = white
      image[29, x] = white
    for y in range(2, 30):
      image[y, 1] = color
      image[y, 30] = color
      image[y, 2] = white
      image[y, 29] = white
    return image

  def generate_skills(files):
    skills = []
    size = 32
    images, names = zip(*files)
    resized = [cv2.resize(img, (size, size)) for img in images]
    files = list(zip(resized, names))
    triangle_mask = np.zeros((size, size), dtype=np.uint8)
    pts = np.array([[0, 0], [31, 0], [16, 16]], np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(triangle_mask, [pts], 255)
    for combo in permutations(files, 3):
      imgs, names = zip(*combo)
      canvas = np.zeros_like(imgs[0])
      for c in range(3):
        canvas[:, :, c] = np.where(triangle_mask == 255, imgs[1][:, :, c], canvas[:, :, c])
      for y in range(size):
        for x in range(size):
          if triangle_mask[y, x] == 0:
            canvas[y, x] = imgs[0][y, x] if x < 16 else imgs[2][y, x]
      skills.append((apply_border(canvas), names))
    return skills

  def find_matching(core, generated_images):
    core = cv2.resize(core, (32, 32))
    scores = [cv2.matchTemplate(core, cv2.resize(g, (32, 32)), cv2.TM_CCOEFF_NORMED)[1] for g in generated_images]
    return np.argmax(scores)

  files = [(cv2.imread(os.path.join(job_folder, f)), f[:-4]) for f in os.listdir(job_folder) if f.endswith('.png')]
  generated = generate_skills(files)
  gen_images, _ = zip(*generated)
  return [generated[find_matching(skill, gen_images)] for skill in core_skills]

def find_combinations(core_skills, target_skills):
  n = len(target_skills)
  skill_index = defaultdict(list)
  for i, core_skill in enumerate(core_skills):
    skill_index[core_skill[0]].append(i)
  all_main = list(skill_index.keys())
  min_case = math.ceil(2 * n / 3)
  max_case = min(2 * n, len(all_main))
  valid_combos = []
  min_len = float('inf')
  for cnt in range(min_case, max_case + 1):
    for main_combo in combinations(all_main, cnt):
      idx_combos = [[]]
      for skill in main_combo:
        idx_combos = [prev + [i] for prev in idx_combos for i in skill_index[skill]]
      for combo in idx_combos:
        counter = Counter()
        for i in combo:
          counter.update(core_skills[i])
        if all(counter[ts] >= 2 for ts in target_skills):
          if len(combo) < min_len:
            min_len = len(combo)
            valid_combos = [combo]
          elif len(combo) == min_len:
            valid_combos.append(combo)
  return valid_combos
