<b>NarrativeHarmBench <b> : <p align="center">
  <b>Evaluating LLM Robustness to Adversarial Narrative Framing in Real-World Events</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/LLM-Safety-blue" />
  <img src="https://img.shields.io/badge/Events-1%2C350-success" />
  <img src="https://img.shields.io/badge/Prompts-4%2C991-orange" />
  <img src="https://img.shields.io/badge/Framing%20Strategies-9-purple" />
</p>

Overview

NarrativeHarmBench is a benchmark for evaluating whether large language models remain factual, balanced, and robust when controversial real-world events are presented through subtle adversarial narrative framing.

Unlike safety benchmarks centered mainly on explicit harmful requests or jailbreaks, NarrativeHarmBench focuses on manipulation such as conspiracy framing, selective evidence, victim-blaming, harm justification, ideological reframing, historical revisionism, and institutional distrust.

<p align="center">
  <img src="./event_category_model_heatmap_flag1.png" width="900" alt="NarrativeHarmBench event-category model performance heatmap">
</p>

<p align="center"><i>Example event-category analysis included in this repository.</i></p>

Benchmark at a Glance
******************************************************************************
Real-world events: 1,350

Retained adversarial prompts: 4,991

Narrative framing strategies: 9

Evaluation dimensions: 10

Models evaluated in the paper: 5
****************************************************************************

Adversarial Framing Strategies

Narrative Challenge

Selective Evidence and Omission

Conspiracy and Hidden-Motive Attribution

Responsibility Reattribution / Victim-Blaming

Harm Justification and Moral Rationalization

Harm Minimization and Impact Denial

Ideological or Partisan Reframing

Historical Revisionism and Comparative Reinterpretation

Institutional Distrust and Legitimacy Challenge

Evaluation

Model responses are scored on a 1–5 scale across:

Factual Accuracy · Contextual Reasoning · Manipulative-Framing Resistance · Misinformation Resistance · Harm-Justification Resistance · Fairness · Neutrality · Safety · Adversarial Robustness · Overall Risk

The paper evaluates:

Gemma 3 12B · Phi-4 · Mistral 7B · Qwen 3.5 9B · DeepSeek-R1-Qwen3-8B

Repository Contents

This repository contains the benchmark data, model responses, LLM-as-a-judge outputs, analysis scripts, human/LLM agreement experiments, and figures used in the project.

Citation

If you use NarrativeHarmBench, please cite:

---

Authors

Mehrnoush Alizade * Krishna Sai Rohith Vadla * Suman Kalyan Maity
Missouri University of Science and Technology

<p align="center">
  <b>NarrativeHarmBench</b><br>
  Real-world events. Adversarial narratives. Robustness beyond jailbreaks.
</p>
