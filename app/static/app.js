"use strict";

const STR = {
  en: {
    title: "Check a leaf for disease",
    lead: "Upload one clear photo of a leaf. AgriScan names the most likely disease and explains why it happens and what you can do.",
    step1: "1. Which crop?",
    step1hint: "(picking it gives better answers)",
    step2: "2. Add a photo",
    choose: "Tap to take or choose a photo",
    change: "Tap to change the photo",
    photoTips: "Best: one leaf, close up, in daylight and in focus.",
    analyse: "Analyse leaf",
    analysing: "Analysing…",
    auto: "Not sure / any",
    footer: "AgriScan is a student portfolio project. The model learned mostly from studio-style photos, so it is less reliable on photos taken in the field, and it knows only 19 conditions of four crops. Use it as a first hint, not as a final answer.",
    credits: "Photos: PlantVillage and PlantDoc (CC BY 4.0). Advice is summarised from university extension services; see the sources under each result.",
    possible: "Most likely",
    healthyBadge: "No known disease seen",
    unsureBadge: "Not sure",
    confidence: "Model confidence",
    confNote: "This is the model's own estimate and can be wrong, especially on field photos.",
    others: "Other possibilities",
    unverified: "(number to be confirmed)",
    peek: "Read about the top guess: {name}",
    peekBadge: "Possible, not confirmed",
    peekNote: "The model is not sure. Treat this as background reading, not a diagnosis, and check with your agriculture office.",
    cause: "Why it happens",
    symptoms: "What to look for",
    conditions: "Weather and conditions that help it",
    spread: "How it spreads",
    actions: "What you can do",
    get_help: "When to get help",
    look_alikes: "Can be confused with",
    tips: "Tips",
    sources: "Sources",
    suggest: "This looks more like {crop}. Try again as {crop}",
    wasRight: "Was this answer right?",
    yes: "Yes",
    no: "No",
    whatWas: "What was it really?",
    dontKnow: "I don't know",
    send: "Send",
    thanks: "Thank you for your feedback.",
    network: "Could not reach the server. Check your connection and try again.",
    timeout: "This is taking too long. Please try again.",
    generic: "Something went wrong. Please try again.",
    err_not_an_image: "That file is not an image we can read. Please choose a JPEG, PNG or WebP photo.",
    err_unsupported_format: "Please upload a JPEG, PNG or WebP photo.",
    err_empty_file: "The file is empty. Please choose another photo.",
    err_too_large: "The photo is too large. Please choose a smaller one.",
    err_too_small: "The photo is too small. Please use a larger, closer photo.",
    err_too_many_pixels: "The photo has too many pixels. Please choose a smaller one.",
    err_corrupt_image: "The photo file is damaged. Please choose another one.",
    err_busy: "The server is busy. Please try again in a moment.",
    err_rate_limited: "Too many requests from you in a short time. Please wait a minute and try again.",
    crops: { tomato: "Tomato", potato: "Potato", pepper: "Pepper", corn: "Corn" },
  },
  bn: {
    title: "পাতার রোগ পরীক্ষা করুন",
    lead: "একটি পাতার পরিষ্কার ছবি আপলোড করুন। AgriScan সবচেয়ে সম্ভাব্য রোগের নাম বলবে এবং কেন হয় ও কী করতে পারেন তা জানাবে।",
    step1: "১. কোন ফসল?",
    step1hint: "(ফসল বেছে দিলে উত্তর আরও ভালো হয়)",
    step2: "২. ছবি দিন",
    choose: "ছবি তুলতে বা বেছে নিতে ট্যাপ করুন",
    change: "ছবি বদলাতে ট্যাপ করুন",
    photoTips: "সবচেয়ে ভালো: একটি পাতা, কাছ থেকে, দিনের আলোয় ও ফোকাসে।",
    analyse: "পাতা বিশ্লেষণ করুন",
    analysing: "বিশ্লেষণ চলছে…",
    auto: "নিশ্চিত নই / যেকোনো",
    footer: "AgriScan একটি শিক্ষার্থীর পোর্টফোলিও প্রকল্প। মডেলটি বেশিরভাগ স্টুডিও-ধাঁচের ছবি থেকে শিখেছে, তাই মাঠে তোলা ছবিতে এটি কম নির্ভরযোগ্য, আর এটি চারটি ফসলের মাত্র ১৯টি অবস্থা চেনে। একে প্রথম ইঙ্গিত হিসেবে ব্যবহার করুন, চূড়ান্ত উত্তর হিসেবে নয়।",
    credits: "ছবি: PlantVillage ও PlantDoc (CC BY 4.0)। পরামর্শ বিশ্ববিদ্যালয়-ভিত্তিক কৃষি সম্প্রসারণ সংস্থার তথ্য থেকে সংক্ষেপে নেওয়া; প্রতিটি ফলাফলের নিচে সূত্র দেখুন।",
    possible: "সবচেয়ে সম্ভাব্য",
    healthyBadge: "জানা কোনো রোগ দেখা যায়নি",
    unsureBadge: "নিশ্চিত নয়",
    confidence: "মডেলের আত্মবিশ্বাস",
    confNote: "এটি মডেলের নিজের অনুমান এবং ভুল হতে পারে, বিশেষ করে মাঠে তোলা ছবিতে।",
    others: "অন্যান্য সম্ভাবনা",
    unverified: "(নম্বরটি নিশ্চিত করা হয়নি)",
    peek: "শীর্ষ অনুমান সম্পর্কে পড়ুন: {name}",
    peekBadge: "সম্ভাব্য, নিশ্চিত নয়",
    peekNote: "মডেল নিশ্চিত নয়। এটিকে রোগনির্ণয় নয়, পটভূমি-তথ্য হিসেবে পড়ুন এবং আপনার কৃষি অফিসের সাথে মিলিয়ে নিন।",
    cause: "কেন হয়",
    symptoms: "কী দেখে চিনবেন",
    conditions: "যেসব আবহাওয়া ও অবস্থায় বাড়ে",
    spread: "কীভাবে ছড়ায়",
    actions: "আপনি কী করতে পারেন",
    get_help: "কখন সাহায্য নেবেন",
    look_alikes: "যার সাথে গুলিয়ে যেতে পারে",
    tips: "পরামর্শ",
    sources: "সূত্র",
    suggest: "এটি দেখতে বেশি {crop}-এর মতো। {crop} হিসেবে আবার চেষ্টা করুন",
    wasRight: "এই উত্তরটি কি সঠিক ছিল?",
    yes: "হ্যাঁ",
    no: "না",
    whatWas: "আসলে এটি কী ছিল?",
    dontKnow: "জানি না",
    send: "পাঠান",
    thanks: "আপনার মতামতের জন্য ধন্যবাদ।",
    network: "সার্ভারের সাথে যোগাযোগ করা যায়নি। ইন্টারনেট দেখে আবার চেষ্টা করুন।",
    timeout: "অনেক সময় লাগছে। আবার চেষ্টা করুন।",
    generic: "কিছু একটা ভুল হয়েছে। আবার চেষ্টা করুন।",
    err_not_an_image: "ফাইলটি পড়ার মতো ছবি নয়। অনুগ্রহ করে JPEG, PNG বা WebP ছবি বেছে নিন।",
    err_unsupported_format: "অনুগ্রহ করে JPEG, PNG বা WebP ছবি দিন।",
    err_empty_file: "ফাইলটি খালি। অন্য ছবি বেছে নিন।",
    err_too_large: "ছবিটি অনেক বড়। ছোট ছবি বেছে নিন।",
    err_too_small: "ছবিটি খুব ছোট। বড় ও কাছ থেকে তোলা ছবি দিন।",
    err_too_many_pixels: "ছবিতে পিক্সেল অনেক বেশি। ছোট ছবি বেছে নিন।",
    err_corrupt_image: "ছবির ফাইলটি নষ্ট। অন্য ছবি বেছে নিন।",
    err_busy: "সার্ভার এখন ব্যস্ত। একটু পরে আবার চেষ্টা করুন।",
    err_rate_limited: "অল্প সময়ে আপনার অনেক অনুরোধ হয়েছে। এক মিনিট অপেক্ষা করে আবার চেষ্টা করুন।",
    crops: { tomato: "টমেটো", potato: "আলু", pepper: "ক্যাপসিকাম", corn: "ভুট্টা" },
  },
};

const CROPS = ["tomato", "potato", "pepper", "corn"];
const SECTIONS = ["cause", "symptoms", "conditions", "spread", "actions", "get_help", "look_alikes"];
const MAX_SIDE = 1600; // photos are shrunk in the browser first: faster upload, same detail for the model
const TIMEOUT_MS = 30000;

const state = {
  lang: "en", crop: "auto", file: null, result: null, classNames: {}, rated: false, peek: null,
};
const PEEK_MIN = 0.5; // "read about the top guess" is offered only when the model is at least this sure
const $ = (id) => document.getElementById(id);
const t = (key) => STR[state.lang][key];

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function safeGet(key) {
  try { return localStorage.getItem(key); } catch { return null; }
}
function safeSet(key, value) {
  try { localStorage.setItem(key, value); } catch { /* private mode: fine */ }
}

/* ---------- language and static text ---------- */

function applyStatic() {
  document.documentElement.lang = state.lang;
  document.querySelectorAll("[data-i18n]").forEach((node) => {
    const key = node.dataset.i18n;
    const target = node.querySelector("[data-i18n]") ? null : node;
    if (key === "step1") {
      node.firstChild.textContent = t("step1") + " ";
      return;
    }
    if (target) target.textContent = t(key);
  });
  const hint = document.querySelector('[data-i18n="step1hint"]');
  if (hint) hint.textContent = t("step1hint");
  $("drop-text").textContent = state.file ? t("change") : t("choose");
  const go = $("go");
  if (!go.dataset.busy) go.textContent = t("analyse");
  document.querySelectorAll(".lang").forEach((b) =>
    b.setAttribute("aria-pressed", String(b.dataset.lang === state.lang)));
  renderChips();
}

async function setLang(lang) {
  state.lang = lang;
  safeSet("agriscan-lang", lang);
  applyStatic();
  $("error").hidden = true;
  if (state.result) await refreshAdvice();
}

/* ---------- crop chips ---------- */

function renderChips() {
  const box = $("crop-chips");
  box.textContent = "";
  for (const value of ["auto", ...CROPS]) {
    const chip = el("button", "chip", value === "auto" ? t("auto") : t("crops")[value]);
    chip.type = "button";
    chip.setAttribute("aria-pressed", String(state.crop === value));
    chip.addEventListener("click", () => {
      state.crop = value;
      renderChips();
    });
    box.appendChild(chip);
  }
}

/* ---------- photo ---------- */

async function shrink(file) {
  try {
    const bitmap = await createImageBitmap(file, { imageOrientation: "from-image" });
    const scale = Math.min(1, MAX_SIDE / Math.max(bitmap.width, bitmap.height));
    if (scale === 1 && file.size < 3 * 1024 * 1024) { bitmap.close(); return file; }
    const canvas = document.createElement("canvas");
    canvas.width = Math.round(bitmap.width * scale);
    canvas.height = Math.round(bitmap.height * scale);
    canvas.getContext("2d").drawImage(bitmap, 0, 0, canvas.width, canvas.height);
    bitmap.close();
    const blob = await new Promise((resolve) => canvas.toBlob(resolve, "image/jpeg", 0.9));
    return blob || file;
  } catch {
    return file; // e.g. a format the browser cannot decode: let the server judge it
  }
}

function onFile() {
  const file = $("file").files[0];
  if (!file) return;
  state.file = file;
  const preview = $("preview");
  if (preview.src) URL.revokeObjectURL(preview.src);
  preview.src = URL.createObjectURL(file);
  preview.hidden = false;
  $("drop-text").textContent = t("change");
  $("go").disabled = false;
  $("error").hidden = true;
}

/* ---------- API ---------- */

async function api(path, options = {}) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), TIMEOUT_MS);
  try {
    const response = await fetch(path, { ...options, signal: controller.signal });
    let body = null;
    try { body = await response.json(); } catch { /* not JSON */ }
    if (!response.ok) {
      const code = body && body.error ? body.error.code : "";
      const known = STR[state.lang]["err_" + code];
      const error = new Error(known || (body && body.error && body.error.message) || t("generic"));
      error.code = code;
      throw error;
    }
    return body;
  } catch (error) {
    if (error.name === "AbortError") throw new Error(t("timeout"));
    if (error instanceof TypeError) throw new Error(t("network"));
    throw error;
  } finally {
    clearTimeout(timer);
  }
}

async function loadClassNames() {
  if (state.classNames[state.lang]) return state.classNames[state.lang];
  try {
    const data = await api("/api/classes?lang=" + state.lang);
    state.classNames[state.lang] = data.classes;
  } catch {
    state.classNames[state.lang] = [];
  }
  return state.classNames[state.lang];
}

async function submit(event) {
  event.preventDefault();
  if (!state.file) return;
  const go = $("go");
  go.disabled = true;
  go.dataset.busy = "1";
  go.textContent = "";
  go.append(el("span", "spinner"), t("analysing"));
  $("error").hidden = true;
  try {
    const blob = await shrink(state.file);
    const form = new FormData();
    form.append("file", blob, "leaf.jpg");
    if (state.crop !== "auto") form.append("crop", state.crop);
    form.append("lang", state.lang);
    state.result = await api("/api/predict", { method: "POST", body: form });
    state.rated = false;
    state.peek = null;
    renderResult();
    $("result").scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (error) {
    $("error").textContent = error.message;
    $("error").hidden = false;
  } finally {
    delete go.dataset.busy;
    go.textContent = t("analyse");
    go.disabled = false;
  }
}

/* ---------- result ---------- */

function fetchDisease(id) {
  return api("/api/diseases/" + encodeURIComponent(id) + "?lang=" + state.lang);
}

async function refreshAdvice() {
  // Language switched while a result is on screen: refetch names and advice, do not re-predict.
  const r = state.result;
  try {
    const names = await loadClassNames();
    const byId = Object.fromEntries(names.map((c) => [c.class_id, c.name]));
    r.predictions.forEach((p) => { p.name = byId[p.class_id] || p.class_id; });
    const id = r.status === "ok" ? r.diagnosis : "unsure";
    r.info = await fetchDisease(id);
    r.lang = state.lang;
    if (state.peek) state.peek.info = await fetchDisease(state.peek.id);
  } catch (error) {
    $("error").textContent = error.message;
    $("error").hidden = false;
  }
  renderResult();
}

function pct(p) {
  return (p * 100).toFixed(p >= 0.995 ? 0 : 1) + "%";
}

function listSection(title, items, open) {
  const box = el("details");
  box.open = !!open;
  box.append(el("summary", "", title));
  const list = el("ul");
  items.forEach((text) => list.append(el("li", "", text)));
  box.append(list);
  return box;
}

function diseaseCard(info) {
  const card = el("div", "card");
  if (info.cause) card.append(listSection(t("cause"), [info.cause], true));
  SECTIONS.slice(1).forEach((key) => {
    if (info[key] && info[key].length) card.append(listSection(t(key), info[key], key === "actions"));
  });
  if (info.chemical_note) card.append(el("p", "notice", info.chemical_note));
  if (info.helpline) {
    const note = info.helpline.verified ? "" : " " + t("unverified");
    card.append(el("p", "helpline", info.helpline.text + note));
  }
  return card;
}

function sourcesCard(info) {
  if (!info.sources || !info.sources.length) return document.createDocumentFragment();
  const card = el("div", "card sources");
  card.append(el("strong", "", t("sources")));
  const list = el("ul");
  info.sources.forEach((src) => {
    const li = el("li");
    const a = el("a", "", src.title);
    a.href = src.url;
    a.target = "_blank";
    a.rel = "noopener noreferrer";
    li.append(a, document.createTextNode(" — " + src.publisher));
    list.append(li);
  });
  card.append(list);
  return card;
}

function renderResult() {
  const r = state.result;
  const box = $("result");
  box.textContent = "";
  box.hidden = false;
  const info = r.info || {};
  const head = el("div", "card");

  if (r.status === "ok") {
    const healthy = info.kind === "healthy";
    head.append(el("span", "badge", healthy ? t("healthyBadge") : t("possible")));
    head.append(el("h2", "dname", (r.predictions[0] && r.predictions[0].name) || r.diagnosis));
    head.append(el("div", "muted", t("confidence") + ": " + pct(r.confidence)));
    const bar = el("div", "bar-bg");
    const fill = el("div", "bar-fg");
    fill.style.width = Math.round(r.confidence * 100) + "%";
    bar.append(fill);
    head.append(bar, el("div", "muted", t("confNote")));
  } else {
    head.append(el("span", "badge warn", t("unsureBadge")));
    head.append(el("p", "", info.message || ""));
    if (r.suggested_crop) {
      const crop = t("crops")[r.suggested_crop] || r.suggested_crop;
      const retry = el("button", "secondary", t("suggest").replaceAll("{crop}", crop));
      retry.type = "button";
      retry.addEventListener("click", () => {
        state.crop = r.suggested_crop;
        renderChips();
        $("form").requestSubmit();
      });
      const row = el("div", "row");
      row.append(retry);
      head.append(row);
    }
    const top = r.predictions[0];
    if (top && top.probability >= PEEK_MIN && !state.peek) {
      const peek = el("button", "secondary", t("peek").replaceAll("{name}", top.name || top.class_id));
      peek.type = "button";
      peek.addEventListener("click", async () => {
        try {
          state.peek = { id: top.class_id, info: await fetchDisease(top.class_id) };
          renderResult();
        } catch (error) {
          $("error").textContent = error.message;
          $("error").hidden = false;
        }
      });
      const row = el("div", "row");
      row.append(peek);
      head.append(row);
    }
  }
  box.append(head);

  if (state.peek && r.status !== "ok" && state.peek.info && state.peek.info.kind === "disease") {
    const note = el("div", "card");
    note.append(el("span", "badge warn", t("peekBadge")));
    note.append(el("h2", "dname", state.peek.info.name));
    note.append(el("p", "muted", t("peekNote")));
    box.append(note, diseaseCard(state.peek.info), sourcesCard(state.peek.info));
  }

  if (info.kind === "disease") {
    box.append(diseaseCard(info));
  } else if (info.kind === "healthy" || info.kind === "unsure") {
    const card = el("div", "card");
    if (info.kind === "healthy" && info.message) card.append(el("p", "", info.message));
    if (info.tips && info.tips.length) card.append(listSection(t("tips"), info.tips, true));
    box.append(card);
  }

  const listed = r.status === "ok" ? r.predictions.slice(1) : r.predictions;
  if (listed.length) {
    const card = el("div", "card");
    card.append(el("strong", "", t("others")));
    const list = el("ul", "top3");
    listed.forEach((p) => {
      const li = el("li");
      li.append(el("span", "", p.name || p.class_id), el("span", "muted", pct(p.probability)));
      list.append(li);
    });
    card.append(list);
    box.append(card);
  }

  if (info.kind === "disease") box.append(sourcesCard(info));

  if (info.disclaimer) box.append(el("p", "muted", info.disclaimer));
  if (r.prediction_id) box.append(feedbackCard(r));
}

function feedbackCard(r) {
  const card = el("div", "card");
  if (state.rated) {
    card.append(el("p", "", t("thanks")));
    return card;
  }
  card.append(el("strong", "", t("wasRight")));
  const row = el("div", "row");
  const yes = el("button", "secondary", t("yes"));
  const no = el("button", "secondary", t("no"));
  yes.type = no.type = "button";
  yes.addEventListener("click", () => sendFeedback(r, "correct", null));
  no.addEventListener("click", async () => {
    const names = await loadClassNames();
    row.textContent = "";
    const select = el("select");
    select.setAttribute("aria-label", t("whatWas"));
    select.append(new Option(t("dontKnow"), ""));
    names.forEach((c) => select.append(new Option(c.name, c.class_id)));
    const send = el("button", "secondary", t("send"));
    send.type = "button";
    send.addEventListener("click", () => sendFeedback(r, "wrong", select.value || null));
    row.append(select, send);
  });
  row.append(yes, no);
  card.append(row);
  return card;
}

async function sendFeedback(r, verdict, trueClass) {
  try {
    const body = { prediction_id: r.prediction_id, verdict };
    if (trueClass) body.true_class = trueClass;
    await api("/api/feedback", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    state.rated = true;
    renderResult();
  } catch (error) {
    $("error").textContent = error.message;
    $("error").hidden = false;
  }
}

/* ---------- start ---------- */

document.addEventListener("DOMContentLoaded", () => {
  const saved = safeGet("agriscan-lang");
  const guess = (navigator.language || "").toLowerCase().startsWith("bn") ? "bn" : "en";
  state.lang = saved === "bn" || saved === "en" ? saved : guess;
  document.querySelectorAll(".lang").forEach((b) =>
    b.addEventListener("click", () => setLang(b.dataset.lang)));
  $("file").addEventListener("change", onFile);
  $("form").addEventListener("submit", submit);
  applyStatic();
});
