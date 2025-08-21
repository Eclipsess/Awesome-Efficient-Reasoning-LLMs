# Awesome-Efficient-Reasoning-LLMs

[![arXiv](https://img.shields.io/badge/arXiv-Stop_Overthinking-b31b1b.svg)](https://arxiv.org/abs/2503.16419)
<!-- [![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)]() <!-- Optional: Link to GitHub repo -->
<!-- [![Last Commit](https://img.shields.io/github/last-commit/<your-username>/<repo-name>)]() <!-- Fill in your repo link -->
<!-- [![Contributions Welcome](https://img.shields.io/badge/Contributions-welcome-blue)]() --> 


## [TMLR 2025] Stop Overthinking: A Survey on Efficient Reasoning for Large Language Models

<!-- omit in toc -->

## 📢 Want to add related papers? Feel free to open a pull request!

## 📢 News
- **August 21, 2025**: Updated.
- **July 14, 2025**: "Stop Overthinking" is accepted by TMLR, Transactions on Machine Learning Research.
- **April 22, 2025**: Updated.
- **March 20, 2025**: We release the first survey for efficient reasoning of LLMs "[Stop Overthinking: A Survey on Efficient Reasoning for Large Language Models](https://arxiv.org/abs/2503.16419)".  
  Feel free to cite, contribute, or open a pull request to add recent related papers!
  

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

* Demystifying Long Chain-of-Thought Reasoning in LLMs [[Paper]](https://arxiv.org/pdf/2502.03373) ![](https://img.shields.io/badge/pdf-2025.02-red)
* O1-Pruner: Length-Harmonizing Fine-Tuning for O1-Like Reasoning Pruning [[Paper]](https://arxiv.org/pdf/2501.12570) ![](https://img.shields.io/badge/pdf-2025.01-red)
* Kimi k1.5: Scaling Reinforcement Learning with LLMs [[Paper]](https://arxiv.org/pdf/2501.12599) ![](https://img.shields.io/badge/pdf-2025.01-red)
* Training Language Models to Reason Efficiently [[Paper]](https://arxiv.org/pdf/2502.04463) ![](https://img.shields.io/badge/pdf-2025.02-red)
* L1: Controlling How Long A Reasoning Model Thinks With Reinforcement Learning [[Paper]](https://www.arxiv.org/pdf/2503.04697) ![](https://img.shields.io/badge/pdf-2025.03-red)
* DAST: Difficulty-Adaptive Slow-Thinking for Large Reasoning Models [[Paper]](https://arxiv.org/pdf/2503.04472) ![](https://img.shields.io/badge/pdf-2025.03-red)
* Optimizing Test-Time Compute via Meta Reinforcement Fine-Tuning [[Paper]](https://arxiv.org/pdf/2503.07572) ![](https://img.shields.io/badge/pdf-2025.03-red)
* HAWKEYE: Efficient Reasoning with Model Collaboration [[Paper]](https://arxiv.org/pdf/2504.00424) ![](https://img.shields.io/badge/pdf-2025.04-red)
* THINKPRUNE: Pruning Long Chain-of-Thought of LLMs via Reinforcement Learning [[Paper]](https://arxiv.org/pdf/2504.01296) ![](https://img.shields.io/badge/pdf-2025.04-red)
* Think When You Need: Self-Adaptive Chain-of-Thought Learning [[Paper]](https://arxiv.org/pdf/2504.03234) ![](https://img.shields.io/badge/pdf-2025.04-red)
* Concise Reasoning via Reinforcement Learning [[Paper]](https://arxiv.org/pdf/2504.05185) ![](https://img.shields.io/badge/pdf-2025.04-red)
* Not All Thoughts are Generated Equal: Efficient LLM Reasoning via Multi-Turn Reinforcement Learning [[Paper]](https://arxiv.org/pdf/2505.11827) ![](https://img.shields.io/badge/pdf-2025.05-red)
* ConciseRL: Conciseness-Guided Reinforcement Learning for Efficient Reasoning Models [[Paper]](https://arxiv.org/pdf/2505.17250) ![](https://img.shields.io/badge/pdf-2025.05-red)
* Scalable Chain of Thoughts via Elastic Reasoning [[Paper]](https://arxiv.org/pdf/2505.05315) ![](https://img.shields.io/badge/pdf-2025.05-red)
* S-GRPO: Early Exit via Reinforcement Learning in Reasoning Models [[Paper]](https://arxiv.org/pdf/2505.07686) ![](https://img.shields.io/badge/pdf-2025.05-red)
* SelfBudgeter: Adaptive Token Allocation for Efficient LLM Reasoning [[Paper]](https://arxiv.org/pdf/2505.11274) ![](https://img.shields.io/badge/pdf-2025.05-red)
* Making Small Language Models Efficient Reasoners: Intervention, Supervision, Reinforcement [[Paper]](https://arxiv.org/pdf/2505.07961) ![](https://img.shields.io/badge/pdf-2025.05-red)
* Efficient RL Training for Reasoning Models via Length-Aware Optimization [[Paper]](https://arxiv.org/pdf/2505.12284) ![](https://img.shields.io/badge/pdf-2025.05-red)
* Optimizing Anytime Reasoning via Budget Relative Policy Optimization [[Paper]](https://arxiv.org/pdf/2505.13438) ![](https://img.shields.io/badge/pdf-2025.05-red)
* Learn to Reason Efficiently with Adaptive Length-based Reward Shaping [[Paper]](https://arxiv.org/pdf/2505.15612) ![](https://img.shields.io/badge/pdf-2025.05-red)
* Incentivizing Dual Process Thinking for Efficient Large Language Model Reasoning [[Paper]](https://arxiv.org/pdf/2505.16315) ![](https://img.shields.io/badge/pdf-2025.05-red)
* LIMOPro: Reasoning Refinement for Efficient and Effective Test-time Scaling [[Paper]](https://arxiv.org/pdf/2505.19187) ![](https://img.shields.io/badge/pdf-2025.05-red)
* Walk Before You Run! Concise LLM Reasoning via Reinforcement Learning [[Paper]](https://arxiv.org/pdf/2505.21178) ![](https://img.shields.io/badge/pdf-2025.05-red)
* Stable Reinforcement Learning for Efficient Reasoning [[Paper]](https://arxiv.org/pdf/2505.18086) ![](https://img.shields.io/badge/pdf-2025.05-red)
* Don't Think Longer, Think Wisely: Optimizing Thinking Dynamics for Large Reasoning Models [[Paper]](https://arxiv.org/pdf/2505.21765) ![](https://img.shields.io/badge/pdf-2025.05-red)
* Thinkless: LLM Learns When to Think. [[Paper]](https://arxiv.org/pdf/2505.13379) ![](https://img.shields.io/badge/pdf-2025.05-red)
* Think Only When You Need with Large Hybrid-Reasoning Models. [[Paper]](https://arxiv.org/pdf/2505.14631) ![](https://img.shields.io/badge/pdf-2025.05-red)
* When to Continue Thinking: Adaptive Thinking Mode Switching for Efficient Reasoning. [[Paper]](https://arxiv.org/pdf/2505.15400) ![](https://img.shields.io/badge/pdf-2025.05-red)
* AdaCoT: Pareto-Optimal Adaptive Chain-of-Thought Triggering via Reinforcement Learning. [[Paper]](https://arxiv.org/pdf/2505.11896) ![](https://img.shields.io/badge/pdf-2025.05-red)
* Learning When to Think: Shaping Adaptive Reasoning in R1-Style Models via Multi-Stage RL. [[Paper]](https://arxiv.org/pdf/2505.10832) ![](https://img.shields.io/badge/pdf-2025.05-red)
* AdaptThink: Reasoning Models Can Learn When to Think. [[Paper]](https://arxiv.org/pdf/2505.13417) ![](https://img.shields.io/badge/pdf-2025.05-red)
* Bingo: Boosting Efficient Reasoning of LLMs via Dynamic and Significance-based Reinforcement Learning [[Paper]](https://arxiv.org/pdf/2506.08125) ![](https://img.shields.io/badge/pdf-2025.06-red)
* How Far Are We from Optimal Reasoning Efficiency? [[Paper]](https://arxiv.org/pdf/2506.07104) ![](https://img.shields.io/badge/pdf-2025.06-red)
* Just Enough Thinking: Efficient Reasoning with Adaptive Length Penalties Reinforcement Learning. [[Paper]](https://arxiv.org/abs/2506.05256) ![](https://img.shields.io/badge/pdf-2025.06-red)
* Fast on the Easy, Deep on the Hard: Efficient Reasoning via Powered Length Penalty. [[Paper]](https://arxiv.org/abs/2506.10446) ![](https://img.shields.io/badge/pdf-2025.06-red)
* Optimizing Length Compression in Large Reasoning Models. [[Paper]](https://arxiv.org/abs/2506.14755) ![](https://img.shields.io/badge/pdf-2025.06-red)
* AdapThink: Adaptive Thinking Preferences for Reasoning Language Model. [[Paper]](https://arxiv.org/abs/2506.18237) ![](https://img.shields.io/badge/pdf-2025.06-red)
* AALC: Large Language Model Efficient Reasoning via Adaptive Accuracy-Length Control. [[Paper]](https://arxiv.org/abs/2506.20160) ![](https://img.shields.io/badge/pdf-2025.06-red)
* Do Thinking Tokens Help or Trap? Towards More Efficient Large Reasoning Model. [[Paper]](https://arxiv.org/abs/2506.23840) ![](https://img.shields.io/badge/pdf-2025.06-red)
* SmartThinker: Learning to Compress and Preserve Reasoning by Step-Level Length Control. [[Paper]](https://arxiv.org/abs/2507.04348) ![](https://img.shields.io/badge/pdf-2025.07-red)
* Reconsidering Overthinking: Penalizing Internal and External Redundancy in CoT Reasoning. [[Paper]](https://arxiv.org/abs/2508.02178) ![](https://img.shields.io/badge/pdf-2025.08-red)
* Train Long, Think Short: Curriculum Learning for Efficient Reasoning. [[Paper]](https://arxiv.org/abs/2508.08940) ![](https://img.shields.io/badge/pdf-2025.08-red)
* Sample More to Think Less: Group Filtered Policy Optimization for Concise Reasoning. [[Paper]](https://arxiv.org/abs/2508.09726) ![](https://img.shields.io/badge/pdf-2025.08-red)
* SABER: Switchable and Balanced Training for Efficient LLM Reasoning.  [[Paper]](https://arxiv.org/abs/2508.10026) ![](https://img.shields.io/badge/pdf-2025.08-red)
* Promoting Efficient Reasoning with Verifiable Stepwise Reward. [[Paper]](https://arxiv.org/abs/2508.10293) ![](https://img.shields.io/badge/pdf-2025.08-red)
* Aware First, Think Less: Dynamic Boundary Self-Awareness Drives Extreme Reasoning Efficiency in Large Language Models. [[Paper]](https://arxiv.org/abs/2508.11582) ![](https://img.shields.io/badge/pdf-2025.08-red)


## Section II: SFT with Variable-Length CoT Data

* TokenSkip: Controllable Chain-of-Thought Compression in LLMs [[Paper]](https://arxiv.org/pdf/2502.12067) ![](https://img.shields.io/badge/pdf-2025.02-red)
* C3oT: Generating Shorter Chain-of-Thought without Compromising Effectiveness [[Paper]](https://arxiv.org/pdf/2412.11664) ![](https://img.shields.io/badge/pdf-2024.12-red)
* CoT-Valve: Length-Compressible Chain-of-Thought Tuning [[Paper]](https://arxiv.org/pdf/2502.09601) ![](https://img.shields.io/badge/pdf-2025.02-red)
* Self-Training Elicits Concise Reasoning in Large Language Models [[Paper]](https://arxiv.org/pdf/2502.20122) ![](https://img.shields.io/badge/pdf-2025.02-red)
* Distilling System 2 into System 1 [[Paper]](https://arxiv.org/pdf/2407.06023) ![](https://img.shields.io/badge/pdf-2024.07-red)
* Can Language Models Learn to Skip Steps? [[Paper]](https://arxiv.org/pdf/2411.01855) ![](https://img.shields.io/badge/pdf-2024.11-red)
* Verbosity-Aware Rationale Reduction: Sentence-Level Rationale Reduction for Efficient and Effective Reasoning. [[Paper]](https://arxiv.org/pdf/2412.21006) ![](https://img.shields.io/badge/pdf-2024.12-red)
* Stepwise Perplexity-Guided Refinement for Efficient Chain-of-Thought Reasoning in Large Language Models [[Paper]](https://arxiv.org/pdf/2502.13260) ![](https://img.shields.io/badge/pdf-2025.02-red)
* Z1: Efficient Test-time Scaling with Code [[Paper]](https://arxiv.org/pdf/2504.00810) ![](https://img.shields.io/badge/pdf-2025.04-red)
* Ada-R1: Hybrid-CoT via Bi-Level Adaptive Reasoning Optimization [[Paper]](https://arxiv.org/pdf/2504.21659) ![](https://img.shields.io/badge/pdf-2025.04-red)
* Long-Short Chain-of-Thought Mixture Supervised Fine-Tuning Eliciting Efficient Reasoning in Large Language Models [[Paper]](https://arxiv.org/pdf/2505.03469) ![](https://img.shields.io/badge/pdf-2025.05-red)
* DRP: Distilled Reasoning Pruning with Skill-aware Step Decomposition for Efficient Large Reasoning Models [[Paper]](https://arxiv.org/pdf/2505.13975) ![](https://img.shields.io/badge/pdf-2025.05-red)
* AutoL2S: Auto Long-Short Reasoning for Efficient Large Language Models [[Paper]](https://arxiv.org/pdf/2505.22662) ![](https://img.shields.io/badge/pdf-2025.05-red)
* Can Pruning Improve Reasoning? Revisiting Long-CoT Compression with Capability in Mind for Better Reasoning [[Paper]](https://arxiv.org/abs/2505.14582) ![](https://img.shields.io/badge/pdf-2025.05-red)
* VeriThinker: Learning to Verify Makes Reasoning Model Efficient [[Paper]](https://arxiv.org/abs/2505.17941) ![](https://img.shields.io/badge/pdf-2025.05-red)
* Assembly of Experts: Linear-time construction of the Chimera LLM variants with emergent and adaptable behaviors [[Paper]](https://arxiv.org/pdf/2506.14794) [[Model Card]](https://huggingface.co/tngtech/DeepSeek-TNG-R1T2-Chimera) [[Free access via OpenRouter]](https://openrouter.ai/tngtech/deepseek-r1t2-chimera:free) ![](https://img.shields.io/badge/pdf-2025.05-red)
* R1-Compress: Long Chain-of-Thought Compression via Chunk Compression and Search [[Paper]](https://arxiv.org/abs/2505.16838) ![](https://img.shields.io/badge/pdf-2025.05-red)
* Not All Tokens Are What You Need In Thinking [[Paper]](https://arxiv.org/abs/2505.17827) ![](https://img.shields.io/badge/pdf-2025.05-red)
* A*-Thought: Efficient Reasoning via Bidirectional Compression for Low-Resource Settings [[Paper]](https://arxiv.org/abs/2505.24550) ![](https://img.shields.io/badge/pdf-2025.05-red)
* ConCISE: Confidence-guided Compression in Step-by-step Efficient Reasoning [[Paper]](https://arxiv.org/abs/2505.04881) ![](https://img.shields.io/badge/pdf-2025.05-red)
* TL;DR: Too Long, Do Re-weighting for Efficient LLM Reasoning Compression [[Paper]](https://arxiv.org/abs/2506.02678) ![](https://img.shields.io/badge/pdf-2025.06-red)
* OThink-R1: Intrinsic Fast/Slow Thinking Mode Switching for Over-Reasoning Mitigation. [[Paper]](https://arxiv.org/pdf/2506.02397) ![](https://img.shields.io/badge/pdf-2025.06-red)
* Causal Sufficiency and Necessity Improves Chain-of-Thought Reasoning [[Paper]](https://arxiv.org/abs/2506.09853) ![](https://img.shields.io/badge/pdf-2025.06-red)
* ReCUT: Balancing Reasoning Length and Accuracy in LLMs via Stepwise Trails and Preference Optimization. [[Paper]](https://arxiv.org/abs/2506.10822) ![](https://img.shields.io/badge/pdf-2025.06-red)
* Compressing Chain-of-Thought in LLMs via Step Entropy. [[Paper]](https://arxiv.org/pdf/2508.03346) ![](https://img.shields.io/badge/pdf-2025.08-red)
* Pruning the Unsurprising: Efficient Code Reasoning via First-Token Surprisal. [[Paper]](https://arxiv.org/pdf/2508.05988) ![](https://img.shields.io/badge/pdf-2025.08-red)

  
## Section III: Compressing Reasoning Steps into Fewer Latent Representation

* Training Large Language Models to Reason in a Continuous Latent Space [[Paper]](https://arxiv.org/pdf/2412.06769) ![](https://img.shields.io/badge/pdf-2024.12-red)
* Compressed Chain of Thought: Efficient Reasoning through Dense Representations [[Paper]](https://arxiv.org/pdf/2412.13171) ![](https://img.shields.io/badge/pdf-2024.12-red)
* Efficient Reasoning with Hidden Thinking (MLLM) [[Paper]](https://arxiv.org/pdf/2501.19201) ![](https://img.shields.io/badge/pdf-2025.01-red)
* SoftCoT: Soft Chain-of-Thought for Efficient Reasoning with LLMs [[Paper]](https://arxiv.org/pdf/2502.12134) ![](https://img.shields.io/badge/pdf-2025.02-red)
* Token Assorted: Mixing Latent and Text Tokens for Improved Language Model Reasoning [[Paper]](https://arxiv.org/pdf/2502.03275) ![](https://img.shields.io/badge/pdf-2025.02-red)
* Reasoning with Latent Thoughts: On the Power of Looped Transformers [[Paper]](https://arxiv.org/pdf/2502.17416) ![](https://img.shields.io/badge/pdf-2025.02-red)
* CODI: Compressing Chain-of-Thought into Continuous Space via Self-Distillation [[Paper]](https://arxiv.org/pdf/2502.21074) ![](https://img.shields.io/badge/pdf-2025.02-red)
* Efficient Reasoning with Hidden Thinking [[Paper]](https://arxiv.org/pdf/2501.19201) ![](https://img.shields.io/badge/pdf-2025.01-red)
* Token Assorted: Mixing Latent and Text Tokens for Improved Language Model Reasoning [[Paper]](https://arxiv.org/pdf/2502.03275) ![](https://img.shields.io/badge/pdf-2025.02-red)
* Back Attention: Understanding and Enhancing Multi-Hop Reasoning in Large Language Models [[Paper]](https://arxiv.org/pdf/2502.10835) ![](https://img.shields.io/badge/pdf-2025.02-red)
* SEAL: Steerable Reasoning Calibration of Large Language Models for Free [[Paper]](https://arxiv.org/pdf/2504.07986) ![](https://img.shields.io/badge/pdf-2025.04-red)
* Think Silently, Think Fast: Dynamic Latent Compression of LLM Reasoning Chains [[Paper]](https://arxiv.org/pdf/2505.16552) ![](https://img.shields.io/badge/pdf-2025.05-red)
* Overclocking LLM Reasoning: Monitoring and Controlling Thinking Path Lengths in LLMs [[Paper]](https://arxiv.org/pdf/2506.07240) ![](https://img.shields.io/badge/pdf-2025.06-red)
* Controlling Thinking Speed in Reasoning Models. [[Paper]](https://arxiv.org/pdf/2507.03704) ![](https://img.shields.io/badge/pdf-2025.07-red)

## Section IV: Dynamic Reasoning Paradigm during Inference

* Efficiently Serving LLM Reasoning Programs with Certaindex [[Paper]](https://arxiv.org/pdf/2412.20993) ![](https://img.shields.io/badge/pdf-2024.12-red)
* When More is Less: Understanding Chain-of-Thought Length in LLMs [[Paper]](https://arxiv.org/pdf/2502.07266) ![](https://img.shields.io/badge/pdf-2025.02-red)
* Sketch-of-Thought: Efficient LLM Reasoning with Adaptive Cognitive-Inspired Sketching [[Paper]](https://arxiv.org/pdf/2503.05179) ![](https://img.shields.io/badge/pdf-2025.03-red)
* Reward-Guided Speculative Decoding for Efficient LLM Reasoning [[Paper]](https://arxiv.org/pdf/2501.19324) ![](https://img.shields.io/badge/pdf-2025.02-red)
* Fast Best-of-N Decoding via Speculative Rejection [[Paper]](https://arxiv.org/pdf/2410.20290) ![](https://img.shields.io/badge/pdf-2024.10-red)
* FastMCTS: A Simple Sampling Strategy for Data Synthesis [[Paper]](https://www.arxiv.org/pdf/2502.11476) ![](https://img.shields.io/badge/pdf-2025.02-red)
* Dynamic Parallel Tree Search for Efficient LLM Reasoning [[Paper]](https://arxiv.org/pdf/2502.16235) ![](https://img.shields.io/badge/pdf-2025.02-red)
* Sampling-Efficient Test-Time Scaling: Self-Estimating the Best-of-N Sampling in Early Decoding [[Paper]](https://arxiv.org/pdf/2503.01422) ![](https://img.shields.io/badge/pdf-2025.03-red)
* LightThinker: Thinking Step-by-Step Compression (training LLMs to compress thoughts into gist tokens) [[Paper]](https://arxiv.org/pdf/2502.15589) ![](https://img.shields.io/badge/pdf-2025.02-red)
* InftyThink: Breaking the Length Limits of Long-Context Reasoning in Large Language Models [[Paper]](https://www.arxiv.org/pdf/2503.06692) ![](https://img.shields.io/badge/pdf-2025.03-red)
* Reasoning Without Self-Doubt: More Efficient Chain-of-Thought Through Certainty Probing [[Paper]](https://openreview.net/pdf?id=wpK4IMJfdX) ![](https://img.shields.io/badge/pdf-2025.03-red)
* SpecReason: Fast and Accurate Inference-Time Compute via Speculative Reasoning [[Paper]](https://arxiv.org/abs/2504.07891) ![](https://img.shields.io/badge/pdf-2025.04-red)
* AdaptiveStep: Automatically Dividing Reasoning Step through Model Confidence [[Paper]](https://arxiv.org/pdf/2502.13943) ![](https://img.shields.io/badge/pdf-2025.02-red)
* Speculative Thinking: Enhancing Small-Model Reasoning with Large Model Guidance at Inference Time [[Paper]](https://arxiv.org/pdf/2504.12329) ![](https://img.shields.io/badge/pdf-2025.04-red)
* Efficient Reasoning for LLMs through Speculative Chain-of-Thought. [[Paper]](https://arxiv.org/pdf/2504.19095) ![](https://img.shields.io/badge/pdf-2025.04-red)
* Can atomic step decomposition enhance the self-structured reasoning of multimodal large models? [[Paper]](https://arxiv.org/pdf/2503.06252) ![](https://img.shields.io/badge/pdf-2025.03-red)
* Think smarter not harder: Adaptive reasoning with inference aware optimization [[Paper]](https://arxiv.org/pdf/2501.17974) ![](https://img.shields.io/badge/pdf-2025.01-red)
* Reasoning Aware Self-Consistency: Leveraging Reasoning Paths for Efficient LLM Sampling [[Paper]](https://arxiv.org/pdf/2408.17017) ![](https://img.shields.io/badge/pdf-2025.02-red)
* Escape Sky-high Cost: Early-stopping Self-Consistency for Multi-step Reasoning [[Paper]](https://arxiv.org/pdf/2401.10480) ![](https://img.shields.io/badge/pdf-2024.01-red)
* Confidence Improves Self-Consistency in LLMs [[Paper]](https://arxiv.org/pdf/2502.06233) ![](https://img.shields.io/badge/pdf-2025.02-red)
* Make every penny count: Difficulty-adaptive self-consistency for cost-efficient reasoning [[Paper]](https://arxiv.org/pdf/2408.13457) ![](https://img.shields.io/badge/pdf-2025.02-red)
* Path-consistency: Prefix enhancement for efficient inference in llm [[Paper]](https://arxiv.org/pdf/2409.01281) ![](https://img.shields.io/badge/pdf-2025.03-red)
* Bridging internal probability and self-consistency for effective and efficient llm reasoning [[Paper]](https://arxiv.org/pdf/2502.00511) ![](https://img.shields.io/badge/pdf-2025.02-red)
* Towards thinking-optimal scaling of test-time compute for llm reasoning [[Paper]](https://arxiv.org/pdf/2502.18080) ![](https://img.shields.io/badge/pdf-2025.02-red)
* Think Deep, Think Fast: Investigating Efficiency of Verifier-free Inference-time-scaling Methods[[Paper]](https://arxiv.org/pdf/2504.14047) ![](https://img.shields.io/badge/pdf-2025.04-red)
* Reasoning models can be effective without thinking [[Paper]](https://arxiv.org/pdf/2504.09858) ![](https://img.shields.io/badge/pdf-2025.04-red)
* Retro-search: Exploring untaken paths for deeper and efficient reasoning [[Paper]](https://arxiv.org/pdf/2504.04383) ![](https://img.shields.io/badge/pdf-2025.04-red)
* Thought manipulation: External thought can be efficient for large reasoning models [[Paper]](https://arxiv.org/pdf/2504.13626) ![](https://img.shields.io/badge/pdf-2025.04-red)
* Sleep-time compute: Beyond inference scaling at test-time [[Paper]](https://arxiv.org/pdf/2504.13171) ![](https://img.shields.io/badge/pdf-2025.04-red)
* Unlocking the capabilities of thought: A reasoning boundary framework to quantify and optimize chain-of-thought [[Paper]](https://arxiv.org/pdf/2410.05695) ![](https://img.shields.io/badge/pdf-2025.04-red)
* THOUGHTTERMINATOR: Benchmarking, Calibrating, and Mitigating Overthinking in Reasoning Models [[Paper]](https://arxiv.org/pdf/2504.13367) ![](https://img.shields.io/badge/pdf-2025.04-red)
* Dynamic Early Exit in Reasoning Models [[Paper]](https://arxiv.org/pdf/2504.15895) ![](https://img.shields.io/badge/pdf-2025.04-red)
* AlphaOne: Reasoning Models Thinking Slow and Fast at Test Time [[Paper]](https://arxiv.org/pdf/2505.24863) ![](https://img.shields.io/badge/pdf-2025.05-red)
* Putting the Value Back in RL: Better Test-Time Scaling by Unifying LLM Reasoners With Verifiers. [[Paper]](https://arxiv.org/pdf/2505.04842) ![](https://img.shields.io/badge/pdf-2025.05-red)
* Guided by Gut: Efficient Test-Time Scaling with Reinforced Intrinsic Confidence. [[Paper]](https://arxiv.org/pdf/2505.20325) ![](https://img.shields.io/badge/pdf-2025.05-red)
* Fractured Chain-of-Thought Reasoning [[Paper]](https://arxiv.org/pdf/2505.12992) ![](https://img.shields.io/badge/pdf-2025.05-red)
* Value-Guided Search for Efficient Chain-of-Thought Reasoning. [[Paper]](https://arxiv.org/pdf/2505.17373) ![](https://img.shields.io/badge/pdf-2025.05-red)
* Don't Overthink it. Preferring Shorter Thinking Chains for Improved LLM Reasoning. [[Paper]](https://arxiv.org/pdf/2505.17813) ![](https://img.shields.io/badge/pdf-2025.05-red)
* First Finish Search: Efficient Test-Time Scaling in Large Language Models [[Paper]](https://arxiv.org/pdf/2505.18149) ![](https://img.shields.io/badge/pdf-2025.05-red)
* Accelerating Large Language Model Reasoning via Speculative Search. [[Paper]](https://arxiv.org/pdf/2505.02865) ![](https://img.shields.io/badge/pdf-2025.05-red)
* FlashThink: An Early Exit Method For Efficient Reasoning [[Paper]](https://arxiv.org/abs/2505.13949) ![](https://img.shields.io/badge/pdf-2025.05-red)
* Reasoning Path Compression: Compressing Generation Trajectories for Efficient LLM Reasoning [[Paper]](https://arxiv.org/abs/2505.13866) ![](https://img.shields.io/badge/pdf-2025.05-red)
* Accelerating Chain-of-Thought Reasoning: When Goal-Gradient Importance Meets Dynamic Skipping [[Paper]](https://arxiv.org/abs/2505.08392) ![](https://img.shields.io/badge/pdf-2025.05-red)
* ThinkLess: A Training-Free Inference-Efficient Method for Reducing Reasoning Redundancy [[Paper]](https://arxiv.org/abs/2505.15684) ![](https://img.shields.io/badge/pdf-2025.05-red)
* Plan and Budget: Effective and Efficient Test-Time Scaling on Large Language Model Reasoning [[Paper]](https://arxiv.org/abs/2505.16122) ![](https://img.shields.io/badge/pdf-2025.05-red)
* TrimR: Verifier-based Training-Free Thinking Compression for Efficient Test-Time Scaling [[Paper]](https://arxiv.org/abs/2505.17155) ![](https://img.shields.io/badge/pdf-2025.05-red)
* CoThink: Token-Efficient Reasoning via Instruct Models Guiding Reasoning Models [[Paper]](https://arxiv.org/abs/2505.22017) ![](https://img.shields.io/badge/pdf-2025.05-red)
* Prolonged Reasoning Is Not All You Need: Certainty-Based Adaptive Routing for Efficient LLM/MLLM Reasoning. [[Paper]](https://arxiv.org/abs/2505.15154) ![](https://img.shields.io/badge/pdf-2025.05-red)
* AlphaOne: Reasoning Models Thinking Slow and Fast at Test Time. [[Paper]](https://arxiv.org/abs/2505.24863) ![](https://img.shields.io/badge/pdf-2025.05-red)
* Every Rollout Counts: Optimal Resource Allocation for Efficient Test-Time Scaling. [[Paper]](https://arxiv.org/pdf/2506.15707) ![](https://img.shields.io/badge/pdf-2025.06-red)
* SPECS: Faster Test-Time Scaling through Speculative Drafts. [[Paper]](https://arxiv.org/pdf/2506.15733) ![](https://img.shields.io/badge/pdf-2025.06-red)
* BEST-Route: Adaptive LLM Routing with Test-Time Optimal Compute. [[Paper]](https://arxiv.org/pdf/2506.22716) ![](https://img.shields.io/badge/pdf-2025.06-red)
* Accelerated Test-Time Scaling with Model-Free Speculative Sampling [[Paper]](https://arxiv.org/abs/2506.04708) ![](https://img.shields.io/badge/pdf-2025.06-red)
* Answer Convergence as a Signal for Early Stopping in Reasoning [[Paper]](https://arxiv.org/abs/2506.02536) ![](https://img.shields.io/badge/pdf-2025.06-red)
* Collaborative LLM Inference via Planning for Efficient Reasoning. [[Paper]](https://arxiv.org/abs/2506.11578) ![](https://img.shields.io/badge/pdf-2025.06-red)
* Wait, We Don't Need to "Wait"! Removing Thinking Tokens Improves Reasoning Efficiency [[Paper]](https://arxiv.org/abs/2506.08343) ![](https://img.shields.io/badge/pdf-2025.06-red)
* Efficient Reasoning Through Suppression of Self-Affirmation Reflections in Large Reasoning Models. [[Paper]](https://arxiv.org/abs/2506.12353) ![](https://img.shields.io/badge/pdf-2025.06-red)
* Steering LLM Thinking with Budget Guidance. [[Paper]](https://arxiv.org/abs/2506.13752) ![](https://img.shields.io/badge/pdf-2025.06-red)
* Exploring and Exploiting the Inherent Efficiency within Large Reasoning Models for Self-Guided Efficiency Enhancement. [[Paper]](https://arxiv.org/abs/2506.15647) ![](https://img.shields.io/badge/pdf-2025.06-red)
* Activation Steering for Chain-of-Thought Compression. [[Paper]](https://arxiv.org/abs/2507.04742) ![](https://img.shields.io/badge/pdf-2025.07-red)
* R-Stitch: Dynamic Trajectory Stitching for Efficient Reasoning. [[Paper]](https://arxiv.org/abs/2507.17307) ![](https://img.shields.io/badge/pdf-2025.07-red)
* MUR: Momentum Uncertainty guided Reasoning for Large Language Models. [[Paper]](https://arxiv.org/abs/2507.14958) ![](https://img.shields.io/badge/pdf-2025.07-red)
* Test-time Prompt Intervention. [[Paper]](https://arxiv.org/abs/2508.02511) ![](https://img.shields.io/badge/pdf-2025.08-red)
* Efficient Reasoning for Large Reasoning Language Models via Certainty-Guided Reflection Suppression. [[Paper]](https://arxiv.org/abs/2508.05337) ![](https://img.shields.io/badge/pdf-2025.08-red)


## Section V: Prompt-Guided Efficient Reasoning

* Token-Budget-Aware LLM Reasoning [[Paper]](https://arxiv.org/pdf/2412.18547) ![](https://img.shields.io/badge/pdf-2025.02-red)
* Chain of Draft: Thinking Faster by Writing Less [[Paper]](https://arxiv.org/pdf/2502.18600) ![](https://img.shields.io/badge/pdf-2025.03-red)
* How Well do LLMs Compress Their Own Chain-of-Thought? A Token Complexity Approach [[Paper]](https://arxiv.org/pdf/2503.01141) ![](https://img.shields.io/badge/pdf-2025.03-red)
* The Benefits of a Concise Chain of Thought on Problem-Solving in Large Language Models [[Paper]](https://arxiv.org/pdf/2401.05618) ![](https://img.shields.io/badge/pdf-2024.10-red)
* Brevity is the soul of sustainability: Characterizing LLM response lengths. [[Paper]](https://arxiv.org/pdf/2506.08686) ![](https://img.shields.io/badge/pdf-2025.06-red)
* PREMISE: Scalable and Strategic Prompt Optimization for Efficient Mathematical Reasoning in Large Models. [[Paper]](https://arxiv.org/pdf/2506.10716) ![](https://img.shields.io/badge/pdf-2025.06-red)
* ConciseHint: Boosting Efficient Reasoning via Continuous Concise Hints during Generation. [[Paper]](https://arxiv.org/pdf/2506.18810) ![](https://img.shields.io/badge/pdf-2025.06-red)

## Section VI: Prompts Attribute-Driven Reasoning Routing
* Claude 3.7 Sonnet and Claude Code [[website]](https://www.anthropic.com/news/claude-3-7-sonnet) ![](https://img.shields.io/badge/html-2025.02-red)
* Sketch-of-Thought: Efficient LLM Reasoning with Adaptive Cognitive-Inspired Sketching [[Paper]](https://arxiv.org/pdf/2503.05179) ![](https://img.shields.io/badge/pdf-2025.03-red)
* Learning to Route LLMs with Confidence Tokens [[Paper]](https://arxiv.org/pdf/2410.13284) ![](https://img.shields.io/badge/pdf-2025.02-red)
* Confident or Seek Stronger: Exploring Uncertainty-Based On-device LLM Routing From Benchmarking to Generalization [[Paper]](https://arxiv.org/pdf/2502.04428) ![](https://img.shields.io/badge/pdf-2025.02-red)
* RouteLLM: Learning to Route LLMs with Preference Data [[Paper]](https://arxiv.org/pdf/2406.18665) ![](https://img.shields.io/badge/pdf-2025.02-red)
* ThinkSwitcher: When to Think Hard, When to Think Fast. [[Paper]](https://arxiv.org/pdf/2505.14183) ![](https://img.shields.io/badge/pdf-2025.05-red)
* Long or short CoT? Investigating Instance-level Switch of Large Reasoning Models. [[Paper]](https://arxiv.org/pdf/2506.04182) ![](https://img.shields.io/badge/pdf-2025.06-red)
* SynapseRoute: An Auto-Route Switching Framework on Dual-State Large Language Model. [[Paper]](https://arxiv.org/pdf/2507.02822) ![](https://img.shields.io/badge/pdf-2025.07-red)

## Section VII: Reasoning Abilities via Efficient Training Data and Model Compression

* LIMO: Less is More for Reasoning [[Paper]](https://arxiv.org/pdf/2502.03387) ![](https://img.shields.io/badge/pdf-2025.02-red)
* s1: Simple test-time scaling [[Paper]](https://arxiv.org/pdf/2501.19393) ![](https://img.shields.io/badge/pdf-2025.03-red)
* S2R: Teaching LLMs to Self-verify and Self-correct via Reinforcement Learning [[Paper]](https://arxiv.org/pdf/2502.12853) ![](https://img.shields.io/badge/pdf-2025.02-red)
* Light-R1: Curriculum SFT, DPO and RL for Long COT from Scratch and Beyond [[Paper]](https://arxiv.org/pdf/2503.10460) ![](https://img.shields.io/badge/pdf-2025.03-red)
* Small Models Struggle to Learn from Strong Reasoners [[Paper]](https://arxiv.org/pdf/2502.12143) ![](https://img.shields.io/badge/pdf-2025.02-red)
* Towards Reasoning Ability of Small Language Models [[Paper]](https://arxiv.org/pdf/2502.11569) ![](https://img.shields.io/badge/pdf-2025.02-red)
* Mixed Distillation Helps Smaller Language Models Reason Better [[Paper]](https://arxiv.org/pdf/2312.10730) ![](https://img.shields.io/badge/pdf-2024.02-red)
* Small language models need strong verifiers to self-correct reasoning [[Paper]](https://arxiv.org/pdf/2404.17140) ![](https://img.shields.io/badge/pdf-2024.06-red)
* Teaching Small Language Models Reasoning through Counterfactual Distillation [[Paper]](https://aclanthology.org/2024.emnlp-main.333.pdf) ![](https://img.shields.io/badge/pdf-2024.11-red)
* Improving Mathematical Reasoning Capabilities of Small Language Models via Feedback-Driven Distillation [[Paper]](https://arxiv.org/pdf/2411.14698) ![](https://img.shields.io/badge/pdf-2024.11-red)
* Probe then retrieve and reason: Distilling probing and reasoning capabilities into smaller language models [[Paper]](https://aclanthology.org/2024.lrec-main.1140.pdf) ![](https://img.shields.io/badge/pdf-2024.05-red)
* Distilling Reasoning Ability from Large Language Models with Adaptive Thinking [[Paper]](https://arxiv.org/pdf/2404.09170) ![](https://img.shields.io/badge/pdf-2024.08-red)
* SKIntern: Internalizing Symbolic Knowledge for Distilling Better CoT Capabilities into Small Language Models [[Paper]](https://arxiv.org/pdf/2409.13183) ![](https://img.shields.io/badge/pdf-2024.12-red)
* TinyR1-32B-Preview: Boosting Accuracy with Branch-Merge Distillation [[Paper]](https://arxiv.org/pdf/2503.04872) ![](https://img.shields.io/badge/pdf-2025.03-red)
* Improving mathematical reasoning capabilities of small language models via feedback-driven distillation [[Paper]](https://arxiv.org/pdf/2411.14698) ![](https://img.shields.io/badge/pdf-2024.11-red)
* Probe then retrieve and reason: Distilling probing and reasoning capabilities into smaller language models [[Paper]](https://arxiv.org/pdf/2212.00193) ![](https://img.shields.io/badge/pdf-2023.05-red)
* TwT: Thinking without Tokens by Habitual Reasoning Distillation with Multi-Teachers’ Guidance [[Paper]](https://arxiv.org/pdf/2503.24198) ![](https://img.shields.io/badge/pdf-2025.03-red)
* When Reasoning Meets Compression: Benchmarking Compressed Large Reasoning Models on Complex Reasoning Tasks [[Paper]](https://arxiv.org/pdf/2504.02010) ![](https://img.shields.io/badge/pdf-2025.04-red)


## Section VIII: Evaluation and Benchmark
* Can 1B LLM Surpass 405B LLM? Rethinking Compute-Optimal Test-Time Scaling [[Paper]](https://arxiv.org/pdf/2502.06703) ![](https://img.shields.io/badge/pdf-2025.02-red)
* The Danger of Overthinking: Examining the Reasoning-Action Dilemma in Agentic Tasks [[Paper]](https://arxiv.org/pdf/2502.08235) ![](https://img.shields.io/badge/pdf-2025.02-red)
* Inference-Time Computations for LLM Reasoning and Planning: A Benchmark and Insights [[Paper]](https://arxiv.org/pdf/2502.12521) ![](https://img.shields.io/badge/pdf-2025.02-red)
* Bag of Tricks: Benchmarking of Jailbreak Attacks on LLMs [[Paper]](https://arxiv.org/pdf/2406.09324) ![](https://img.shields.io/badge/pdf-2024.11-red)
* The Impact of Reasoning Step Length on Large Language Models [[Paper]](https://arxiv.org/html/2401.04925v3) ![](https://img.shields.io/badge/pdf-2024.01-red)
* S1-bench: A simple benchmark for evaluating system 1 thinking capability of large reasoning models [[Paper]](https://arxiv.org/pdf/2504.10368) ![](https://img.shields.io/badge/pdf-2025.04-red)
* When reasoning meets compression: Benchmarking compressed large reasoning models on complex reasoning tasks. [[Paper]](https://arxiv.org/pdf/2504.02010) ![](https://img.shields.io/badge/pdf-2025.04-red)
* Quantization Hurts Reasoning? An Empirical Study on Quantized Reasoning Models [[Paper]](https://arxiv.org/pdf/2504.04823) ![](https://img.shields.io/badge/pdf-2025.04-red)
* A Technical Study into 0.5B Reasoning Language Models. [[Paper]](https://arxiv.org/pdf/2506.13404) ![](https://img.shields.io/badge/pdf-2025.06-red)



## Citation
If you find this work useful, please cite us.
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
> 🧩 *Layout inspired by [zzli2022/Awesome-System2-Reasoning-LLM](https://github.com/zzli2022/Awesome-System2-Reasoning-LLM) and the latest works are referred to as [hemingkx/Awesome-Efficient-Reasoning](https://github.com/hemingkx/Awesome-Efficient-Reasoning). Many thanks for the great structure!*
