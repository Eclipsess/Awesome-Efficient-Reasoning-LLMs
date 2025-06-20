# Awesome-Efficient-Reasoning-LLMs

[![arXiv](https://img.shields.io/badge/arXiv-Stop_Overthinking-b31b1b.svg)](https://arxiv.org/abs/2503.16419)
<!-- [![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)]() <!-- Optional: Link to GitHub repo -->
<!-- [![Last Commit](https://img.shields.io/github/last-commit/<your-username>/<repo-name>)]() <!-- Fill in your repo link -->
<!-- [![Contributions Welcome](https://img.shields.io/badge/Contributions-welcome-blue)]() --> 


<!-- omit in toc -->

## 📢 Want to add related papers? Feel free to open a pull request!

## 📢 News
- **March 20, 2025**: We release the first survey for efficient reasoning of LLMs "[Stop Overthinking: A Survey on Efficient Reasoning for Large Language Models](https://arxiv.org/abs/2503.16419)".  
  Feel free to cite, contribute, or open a pull request to add recent related papers!
- **April 22, 2025**: Updated.
  

<!-- omit in toc -->
![Pipeline](./figs/pipeline1.png)

In this paper, we present the first structured survey that systematically investigates and organizes the current progress in achieving **efficient reasoning in LLMs**.

## 📊 Taxonomy

Below is a taxonomy graph summarizing the current landscape of efficient reasoning research for LLMs:

![Taxonomy](./figs/taxonomy.png)

---

<!-- omit in toc -->
## 📚 Table of Contents

- [Awesome-Efficient-Reasoning-LLM](#awesome-efficient-reasoning-llm)
  - **Model-based Efficient Reasoning**
    - [Section I: RL with Length Reward Design](#section-i-rl-with-length-reward-design)
    - [Section II: SFT with Variable-Length CoT Data](#section-ii-sft-with-variable-length-cot-data)
  - **Reasoning Output-based Efficient Reasoning**
    - [Section III: Compressing Reasoning Steps into Fewer Latent Representation](#section-iii-compressing-reasoning-steps-into-fewer-latent-representation)
    - [Section IV: Dynamic Reasoning Paradigm during Inference](#section-iv-dynamic-reasoning-paradigm-during-inference)
  - **Input Prompt-based Efficient Reasoning**
    - [Section V: Prompt-Guided Efficient Reasoning](#section-v-prompt-guided-efficient-reasoning)
    - [Section VI: Prompts Attribute-Driven Reasoning Routing](#section-vi-prompts-attribute-driven-reasoning-routing)
  - **Reasoning Abilities with Efficient Data and Small Language Models**
    - [Section VII: Reasoning Abilities via Efficient Training Data and Model Compression](#section-vii-reasoning-abilities-via-efficient-training-data-and-model-compression)
  - **Evaluation and Benchmark**
    - [Section VIII: Evaluation and Benchmark](#section-viii-evaluation-and-benchmark)


---

<!--[[Paper]](pdf LINK) ![](https://img.shields.io/badge/pdf-< TIME >-red)-->

"(.)" stands for "To Be Updated" in the survey paper.

## Section I:  RL with Length Reward Design

* Demystifying Long Chain-of-Thought Reasoning in LLMs [![Paper](https://img.shields.io/badge/pdf-2025.02-red)](https://arxiv.org/pdf/2502.03373)
* O1-Pruner: Length-Harmonizing Fine-Tuning for O1-Like Reasoning Pruning [![Paper](https://img.shields.io/badge/pdf-2025.01-red)](https://arxiv.org/pdf/2501.12570)
* Kimi k1.5: Scaling Reinforcement Learning with LLMs [![Paper](https://img.shields.io/badge/pdf-2025.01-red)](https://arxiv.org/pdf/2501.12599)
* Training Language Models to Reason Efficiently [![Paper](https://img.shields.io/badge/pdf-2025.02-red)](https://arxiv.org/pdf/2502.04463)
* L1: Controlling How Long A Reasoning Model Thinks With Reinforcement Learning [![Paper](https://img.shields.io/badge/pdf-2025.03-red)](https://www.arxiv.org/pdf/2503.04697)
* DAST: Difficulty-Adaptive Slow-Thinking for Large Reasoning Models [![Paper](https://img.shields.io/badge/pdf-2025.03-red)](https://arxiv.org/pdf/2503.04472)
* Optimizing Test-Time Compute via Meta Reinforcement Fine-Tuning [![Paper](https://img.shields.io/badge/pdf-2025.03-red)](https://arxiv.org/pdf/2503.07572)
* HAWKEYE: Efficient Reasoning with Model Collaboration [![Paper](https://img.shields.io/badge/pdf-2025.04-red)](https://arxiv.org/pdf/2504.00424)
* THINKPRUNE: Pruning Long Chain-of-Thought of LLMs via Reinforcement Learning [![Paper](https://img.shields.io/badge/pdf-2025.04-red)](https://arxiv.org/pdf/2504.01296)
* Think When You Need: Self-Adaptive Chain-of-Thought Learning [![Paper](https://img.shields.io/badge/pdf-2025.04-red)](https://arxiv.org/pdf/2504.03234)
* Not All Thoughts are Generated Equal: Efficient LLM Reasoning via Multi-Turn Reinforcement Learning (.) [![Paper](https://img.shields.io/badge/pdf-2025.05-red)](https://arxiv.org/pdf/2505.11827)
* ConciseRL: Conciseness-Guided Reinforcement Learning for Efficient Reasoning Models (.) [![Paper](https://img.shields.io/badge/pdf-2025.05-red)](https://arxiv.org/pdf/2505.17250)
* Bingo: Boosting Efficient Reasoning of LLMs via Dynamic and Significance-based Reinforcement Learning (.) [![Paper](https://img.shields.io/badge/pdf-2025.06-red)](https://arxiv.org/pdf/2506.08125)

## Section II: SFT with Variable-Length CoT Data

* TokenSkip: Controllable Chain-of-Thought Compression in LLMs [![Paper](https://img.shields.io/badge/pdf-2025.02-red)](https://arxiv.org/pdf/2502.12067)
* C3oT: Generating Shorter Chain-of-Thought without Compromising Effectiveness [![Paper](https://img.shields.io/badge/pdf-2024.12-red)](https://arxiv.org/pdf/2412.11664)
* CoT-Valve: Length-Compressible Chain-of-Thought Tuning [![Paper](https://img.shields.io/badge/pdf-2025.02-red)](https://arxiv.org/pdf/2502.09601)
* Self-Training Elicits Concise Reasoning in Large Language Models [![Paper](https://img.shields.io/badge/pdf-2025.02-red)](https://arxiv.org/pdf/2502.20122)
* Distilling System 2 into System 1 [![Paper](https://img.shields.io/badge/pdf-2024.07-red)](https://arxiv.org/pdf/2407.06023)
* Can Language Models Learn to Skip Steps? [![Paper](https://img.shields.io/badge/pdf-2024.11-red)](https://arxiv.org/pdf/2411.01855)
* Stepwise Perplexity-Guided Refinement for Efficient Chain-of-Thought Reasoning in Large Language Models [![Paper](https://img.shields.io/badge/pdf-2025.02-red)](https://arxiv.org/pdf/2502.13260)
* Z1: Efficient Test-time Scaling with Code [![Paper](https://img.shields.io/badge/pdf-2025.04-red)](https://arxiv.org/pdf/2504.00810)
* Long-Short Chain-of-Thought Mixture Supervised Fine-Tuning Eliciting Efficient Reasoning in Large Language Models (.) [![Paper](https://img.shields.io/badge/pdf-2025.05-red)](https://arxiv.org/pdf/2505.03469)
* DRP: Distilled Reasoning Pruning with Skill-aware Step Decomposition for Efficient Large Reasoning Models (.) [![Paper](https://img.shields.io/badge/pdf-2025.05-red)](https://arxiv.org/pdf/2505.13975)
* Not All Thoughts are Generated Equal: Efficient LLM Reasoning via Multi-Turn Reinforcement Learning (.) [![Paper](https://img.shields.io/badge/pdf-2025.05-red)](https://arxiv.org/pdf/2505.11827)
* AutoL2S: Auto Long-Short Reasoning for Efficient Large Language Models (.) [![Paper](https://img.shields.io/badge/pdf-2025.05-red)](https://arxiv.org/pdf/2505.22662)

## Section III: Compressing Reasoning Steps into Fewer Latent Representation

* Training Large Language Models to Reason in a Continuous Latent Space [![Paper](https://img.shields.io/badge/pdf-2024.12-red)](https://arxiv.org/pdf/2412.06769)
* Compressed Chain of Thought: Efficient Reasoning through Dense Representations [![Paper](https://img.shields.io/badge/pdf-2024.12-red)](https://arxiv.org/pdf/2412.13171)
* Efficient Reasoning with Hidden Thinking (MLLM) [![Paper](https://img.shields.io/badge/pdf-2025.01-red)](https://arxiv.org/pdf/2501.19201)
* SoftCoT: Soft Chain-of-Thought for Efficient Reasoning with LLMs [![Paper](https://img.shields.io/badge/pdf-2025.02-red)](https://arxiv.org/pdf/2502.12134)
* Token Assorted: Mixing Latent and Text Tokens for Improved Language Model Reasoning [![Paper](https://img.shields.io/badge/pdf-2025.02-red)](https://arxiv.org/pdf/2502.03275)
* Reasoning with Latent Thoughts: On the Power of Looped Transformers [![Paper](https://img.shields.io/badge/pdf-2025.02-red)](https://arxiv.org/pdf/2502.17416)
* CODI: Compressing Chain-of-Thought into Continuous Space via Self-Distillation [![Paper](https://img.shields.io/badge/pdf-2025.02-red)](https://arxiv.org/pdf/2502.21074)
* Efficient Reasoning with Hidden Thinking [![Paper](https://img.shields.io/badge/pdf-2025.01-red)](https://arxiv.org/pdf/2501.19201)
* Token Assorted: Mixing Latent and Text Tokens for Improved Language Model Reasoning [![Paper](https://img.shields.io/badge/pdf-2025.02-red)](https://arxiv.org/pdf/2502.03275)
* Back Attention: Understanding and Enhancing Multi-Hop Reasoning in Large Language Models (.) [![Paper](https://img.shields.io/badge/pdf-2025.02-red)](https://arxiv.org/pdf/2502.10835)
* Think Silently, Think Fast: Dynamic Latent Compression of LLM Reasoning Chains (.) [![Paper](https://img.shields.io/badge/pdf-2025.05-red)](https://arxiv.org/pdf/2505.16552)


## Section IV: Dynamic Reasoning Paradigm during Inference

* Efficiently Serving LLM Reasoning Programs with Certaindex [![Paper](https://img.shields.io/badge/pdf-2024.12-red)](https://arxiv.org/pdf/2412.20993)
* When More is Less: Understanding Chain-of-Thought Length in LLMs [![Paper](https://img.shields.io/badge/pdf-2025.02-red)](https://arxiv.org/pdf/2502.07266)
* Sketch-of-Thought: Efficient LLM Reasoning with Adaptive Cognitive-Inspired Sketching [![Paper](https://img.shields.io/badge/pdf-2025.03-red)](https://arxiv.org/pdf/2503.05179)
* Reward-Guided Speculative Decoding for Efficient LLM Reasoning [![Paper](https://img.shields.io/badge/pdf-2025.02-red)](https://arxiv.org/pdf/2501.19324)
* Fast Best-of-N Decoding via Speculative Rejection [![Paper](https://img.shields.io/badge/pdf-2024.10-red)](https://arxiv.org/pdf/2410.20290)
* FastMCTS: A Simple Sampling Strategy for Data Synthesis [![Paper](https://img.shields.io/badge/pdf-2025.02-red)](https://www.arxiv.org/pdf/2502.11476)
* Dynamic Parallel Tree Search for Efficient LLM Reasoning [![Paper](https://img.shields.io/badge/pdf-2025.02-red)](https://arxiv.org/pdf/2502.16235)
* Sampling-Efficient Test-Time Scaling: Self-Estimating the Best-of-N Sampling in Early Decoding [![Paper](https://img.shields.io/badge/pdf-2025.03-red)](https://arxiv.org/pdf/2503.01422)
* LightThinker: Thinking Step-by-Step Compression (training LLMs to compress thoughts into gist tokens) [![Paper](https://img.shields.io/badge/pdf-2025.02-red)](https://arxiv.org/pdf/2502.15589)
* InftyThink: Breaking the Length Limits of Long-Context Reasoning in Large Language Models [![Paper](https://img.shields.io/badge/pdf-2025.03-red)](https://www.arxiv.org/pdf/2503.06692)
* Reasoning Without Self-Doubt: More Efficient Chain-of-Thought Through Certainty Probing [![Paper](https://img.shields.io/badge/pdf-2025.03-red)](https://openreview.net/pdf?id=wpK4IMJfdX)
* SpecReason: Fast and Accurate Inference-Time Compute via Speculative Reasoning [![Paper](https://img.shields.io/badge/pdf-2025.04-red)](https://arxiv.org/abs/2504.07891)
* AdaptiveStep: Automatically Dividing Reasoning Step through Model Confidence [![Paper](https://img.shields.io/badge/pdf-2025.02-red)](https://arxiv.org/pdf/2502.13943)
* Speculative Thinking: Enhancing Small-Model Reasoning with Large Model Guidance at Inference Time [![Paper](https://img.shields.io/badge/pdf-2025.04-red)](https://arxiv.org/pdf/2504.12329)
* Can atomic step decomposition enhance the self-structured reasoning of multimodal large models? [![Paper](https://img.shields.io/badge/pdf-2025.03-red)](https://arxiv.org/pdf/2503.06252)
* Think smarter not harder: Adaptive reasoning with inference aware optimization [![Paper](https://img.shields.io/badge/pdf-2025.01-red)](https://arxiv.org/pdf/2501.17974)
* Reasoning Aware Self-Consistency: Leveraging Reasoning Paths for Efficient LLM Sampling [![Paper](https://img.shields.io/badge/pdf-2025.02-red)](https://arxiv.org/pdf/2408.17017)
* Escape Sky-high Cost: Early-stopping Self-Consistency for Multi-step Reasoning [![Paper](https://img.shields.io/badge/pdf-2024.01-red)](https://arxiv.org/pdf/2401.10480)
* Confidence Improves Self-Consistency in LLMs [![Paper](https://img.shields.io/badge/pdf-2025.02-red)](https://arxiv.org/pdf/2502.06233)
* Make every penny count: Difficulty-adaptive self-consistency for cost-efficient reasoning [![Paper](https://img.shields.io/badge/pdf-2025.02-red)](https://arxiv.org/pdf/2408.13457)
* Path-consistency: Prefix enhancement for efficient inference in llm [![Paper](https://img.shields.io/badge/pdf-2025.03-red)](https://arxiv.org/pdf/2409.01281)
* Bridging internal probability and self-consistency for effective and efficient llm reasoning [![Paper](https://img.shields.io/badge/pdf-2025.02-red)](https://arxiv.org/pdf/2502.00511)
* Towards thinking-optimal scaling of test-time compute for llm reasoning [![Paper](https://img.shields.io/badge/pdf-2025.02-red)](https://arxiv.org/pdf/2502.18080)
* Think Deep, Think Fast: Investigating Efficiency of Verifier-free Inference-time-scaling Methods[![Paper](https://img.shields.io/badge/pdf-2025.04-red)](https://arxiv.org/pdf/2504.14047)
* Reasoning models can be effective without thinking [![Paper](https://img.shields.io/badge/pdf-2025.04-red)](https://arxiv.org/pdf/2504.09858)
* Retro-search: Exploring untaken paths for deeper and efficient reasoning [![Paper](https://img.shields.io/badge/pdf-2025.04-red)](https://arxiv.org/pdf/2504.04383)
* Thought manipulation: External thought can be efficient for large reasoning models [![Paper](https://img.shields.io/badge/pdf-2025.04-red)](https://arxiv.org/pdf/2504.13626)
* Sleep-time compute: Beyond inference scaling at test-time [![Paper](https://img.shields.io/badge/pdf-2025.04-red)](https://arxiv.org/pdf/2504.13171)
* Unlocking the capabilities of thought: A reasoning boundary framework to quantify and optimize chain-of-thought [![Paper](https://img.shields.io/badge/pdf-2025.04-red)](https://arxiv.org/pdf/2410.05695)
* THOUGHTTERMINATOR: Benchmarking, Calibrating, and Mitigating Overthinking in Reasoning Models [![Paper](https://img.shields.io/badge/pdf-2025.04-red)](https://arxiv.org/pdf/2504.13367)
* Dynamic Early Exit in Reasoning Models [![Paper](https://img.shields.io/badge/pdf-2025.04-red)](https://arxiv.org/pdf/2504.15895)
* Accelerated Test-Time Scaling with Model-Free Speculative Sampling (.) [![Paper](https://img.shields.io/badge/pdf-2025.06-red)](https://arxiv.org/abs/2506.04708)

## Section V: Prompt-Guided Efficient Reasoning

* Token-Budget-Aware LLM Reasoning [![Paper](https://img.shields.io/badge/pdf-2025.02-red)](https://arxiv.org/pdf/2412.18547)
* Chain of Draft: Thinking Faster by Writing Less [![Paper](https://img.shields.io/badge/pdf-2025.03-red)](https://arxiv.org/pdf/2502.18600)
* How Well do LLMs Compress Their Own Chain-of-Thought? A Token Complexity Approach [![Paper](https://img.shields.io/badge/pdf-2025.03-red)](https://arxiv.org/pdf/2503.01141)
* The Benefits of a Concise Chain of Thought on Problem-Solving in Large Language Models [![Paper](https://img.shields.io/badge/pdf-2024.10-red)](https://arxiv.org/pdf/2401.05618)

## Section VI: Prompts Attribute-Driven Reasoning Routing
* Claude 3.7 Sonnet and Claude Code [![website](https://img.shields.io/badge/html-2025.02-red)](https://www.anthropic.com/news/claude-3-7-sonnet)
* Sketch-of-Thought: Efficient LLM Reasoning with Adaptive Cognitive-Inspired Sketching [![Paper](https://img.shields.io/badge/pdf-2025.03-red)](https://arxiv.org/pdf/2503.05179)
* Learning to Route LLMs with Confidence Tokens [![Paper](https://img.shields.io/badge/pdf-2025.02-red)](https://arxiv.org/pdf/2410.13284)
* Confident or Seek Stronger: Exploring Uncertainty-Based On-device LLM Routing From Benchmarking to Generalization [![Paper](https://img.shields.io/badge/pdf-2025.02-red)](https://arxiv.org/pdf/2502.04428)
* RouteLLM: Learning to Route LLMs with Preference Data [![Paper](https://img.shields.io/badge/pdf-2025.02-red)](https://arxiv.org/pdf/2406.18665)

## Section VII: Reasoning Abilities via Efficient Training Data and Model Compression

* LIMO: Less is More for Reasoning [![Paper](https://img.shields.io/badge/pdf-2025.02-red)](https://arxiv.org/pdf/2502.03387)
* s1: Simple test-time scaling [![Paper](https://img.shields.io/badge/pdf-2025.03-red)](https://arxiv.org/pdf/2501.19393)
* S2R: Teaching LLMs to Self-verify and Self-correct via Reinforcement Learning [![Paper]((https://img.shields.io/badge/pdf-2025.02-red))](https://arxiv.org/pdf/2502.12853)
* Light-R1: Curriculum SFT, DPO and RL for Long COT from Scratch and Beyond [![Paper](https://img.shields.io/badge/pdf-2025.03-red)](https://arxiv.org/pdf/2503.10460)
* Small Models Struggle to Learn from Strong Reasoners [![Paper](https://img.shields.io/badge/pdf-2025.02-red)](https://arxiv.org/pdf/2502.12143)
* Towards Reasoning Ability of Small Language Models [![Paper](https://img.shields.io/badge/pdf-2025.02-red)](https://arxiv.org/pdf/2502.11569)
* Mixed Distillation Helps Smaller Language Models Reason Better [![Paper](https://img.shields.io/badge/pdf-2024.02-red)](https://arxiv.org/pdf/2312.10730)
* Small language models need strong verifiers to self-correct reasoning [![Paper](https://img.shields.io/badge/pdf-2024.06-red)](https://arxiv.org/pdf/2404.17140)
* Teaching Small Language Models Reasoning through Counterfactual Distillation [![Paper](https://img.shields.io/badge/pdf-2024.11-red)](https://aclanthology.org/2024.emnlp-main.333.pdf)
* Improving Mathematical Reasoning Capabilities of Small Language Models via Feedback-Driven Distillation [![Paper](https://img.shields.io/badge/pdf-2024.11-red)](https://arxiv.org/pdf/2411.14698)
* Probe then retrieve and reason: Distilling probing and reasoning capabilities into smaller language models [![Paper](https://img.shields.io/badge/pdf-2024.05-red)](https://aclanthology.org/2024.lrec-main.1140.pdf)
* Distilling Reasoning Ability from Large Language Models with Adaptive Thinking [![Paper](https://img.shields.io/badge/pdf-2024.08-red)](https://arxiv.org/pdf/2404.09170)
* SKIntern: Internalizing Symbolic Knowledge for Distilling Better CoT Capabilities into Small Language Models [![Paper](https://img.shields.io/badge/pdf-2024.12-red)](https://arxiv.org/pdf/2409.13183)
* TinyR1-32B-Preview: Boosting Accuracy with Branch-Merge Distillation [![Paper](https://img.shields.io/badge/pdf-2025.03-red)](https://arxiv.org/pdf/2503.04872)
* Improving mathematical reasoning capabilities of small language models via feedback-driven distillation [![Paper](https://img.shields.io/badge/pdf-2024.11-red)](https://arxiv.org/pdf/2411.14698)
* Probe then retrieve and reason: Distilling probing and reasoning capabilities into smaller language models [![Paper](https://img.shields.io/badge/pdf-2023.05-red)](https://arxiv.org/pdf/2212.00193)
* TwT: Thinking without Tokens by Habitual Reasoning Distillation with Multi-Teachers’ Guidance [![Paper](https://img.shields.io/badge/pdf-2025.03-red)](https://arxiv.org/pdf/2503.24198)
* When Reasoning Meets Compression: Benchmarking Compressed Large Reasoning Models on Complex Reasoning Tasks [![Paper](https://img.shields.io/badge/pdf-2025.04-red)](https://arxiv.org/pdf/2504.02010)

## Section VIII: Evaluation and Benchmark
* Can 1B LLM Surpass 405B LLM? Rethinking Compute-Optimal Test-Time Scaling [![Paper](https://img.shields.io/badge/pdf-2025.02-red)](https://arxiv.org/pdf/2502.06703)
* The Danger of Overthinking: Examining the Reasoning-Action Dilemma in Agentic Tasks [![Paper](https://img.shields.io/badge/pdf-2025.02-red)](https://arxiv.org/pdf/2502.08235)
* Inference-Time Computations for LLM Reasoning and Planning: A Benchmark and Insights [![Paper](https://img.shields.io/badge/pdf-2025.02-red)](https://arxiv.org/pdf/2502.12521)
* Bag of Tricks: Benchmarking of Jailbreak Attacks on LLMs [![Paper](https://img.shields.io/badge/pdf-2024.11-red)](https://arxiv.org/pdf/2406.09324)
* The Impact of Reasoning Step Length on Large Language Models [![Paper](https://img.shields.io/badge/pdf-2024.01-red)](https://arxiv.org/html/2401.04925v3)
* S1-bench: A simple benchmark for evaluating system 1 thinking capability of large reasoning models [![Paper](https://img.shields.io/badge/pdf-2025.04-red)](https://arxiv.org/pdf/2504.10368)
* When reasoning meets compression: Benchmarking compressed large reasoning models on complex reasoning tasks. [![Paper](https://img.shields.io/badge/pdf-2025.04-red)](https://arxiv.org/pdf/2504.02010)
* Quantization Hurts Reasoning? An Empirical Study on Quantized Reasoning Models [![Paper](https://img.shields.io/badge/pdf-2025.04-red)](https://arxiv.org/pdf/2504.04823)




## Citation
If you find this work useful, welcome to cite us.
```bib
@misc{sui2025stopoverthinkingsurveyefficient,
      title={Stop Overthinking: A Survey on Efficient Reasoning for Large Language Models}, 
      author={Yang Sui and Yu-Neng Chuang and Guanchu Wang and Jiamu Zhang and Tianyi Zhang and Jiayi Yuan and Hongyi Liu and Andrew Wen and Shaochen Zhong and Hanjie Chen and Xia Hu},
      year={2025},
      eprint={2503.16419},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2503.16419}, 
}
```

## Acknowledgment
> 🧩 *Layout inspired by [zzli2022/Awesome-System2-Reasoning-LLM](https://github.com/zzli2022/Awesome-System2-Reasoning-LLM). Many thanks for the great structure!*
