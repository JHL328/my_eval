import os
import json
import re
from tqdm import tqdm
from vllm import LLM, SamplingParams
import csv
import argparse

# 5-shot CoT fewshot examples from gpqa_diamond train split (前5条)
FEWSHOT_EXAMPLES = [
    {
        "question": "A large gene has dozens of exons, of which the central ones code for folded triple helical repeats that connect the cytoskeleton with sarcolemma and extracellular space. Each exon usually codes for one folded triple alpha helix. The most common mutations of the gene are central exon deletions that create out-of-frame peptides and progressive degenerative organ waste. A solution is to deliver a Morpholino that recognizes the 5' end of the out-of-frame exon in pre-mRNA. The molecule prevents binding of the spliceosome and creates exon skipping and in-frame joining. Several missing exons are well tolerated by an organism. Which structure below is not involved in the proposed therapy?\n\nA. R-loops.\nB. lariat.\nC. polyA tail.\nD. antisense.\n\nPlease reason step-by-step and put your choice letter without any other text with \\boxed{} in the end.",
        "target": "The text describes the dystrophin gene and the FDA-approved oligonucleotide therapy that causes exon skipping by creating a functional, albeit shorter, dystrophin protein. Morpholino is bound to the pre-mRNA in an antisense orientation. Every splicing mechanism creates the lariat molecule that is circular with a 3' tail and soon degraded. The spliced RNA is polyadenylated at the 3' end. R-loops are triple helix of DNA and the pre-mRNA and a consequence of the RNA transcription, not splicing and RNA maturation. Therefore, the answer is \\boxed{A}."
    },
    {
        "question": "Two quantum states with energies E1 and E2 have a lifetime of 10^-9 sec and 10^-8 sec, respectively. We want to clearly distinguish these two energy levels. Which one of the following options could be their energy difference so that they can be clearly resolved?\n\nA. 10^-9 eV\nB. 10^-8 eV\nC. 10^-11 eV\nD. 10^-4 eV\n\nPlease reason step-by-step and put your choice letter without any other text with \\boxed{} in the end.",
        "target": "According to the uncertainty principle, Delta E* Delta t=hbar/2. Delta t is the lifetime and Delta E is the width of the energy level. With Delta t=10^-9 s==> Delta E1= 3.3 10^-7 ev. And Delta t=10^-8 s gives Delta E2=3.3*10^-8 eV. Therefore, the energy difference between the two states must be significantly greater than 10^-7 ev. So the answer is \\boxed{D}."
    },
    {
        "question": "How many of the following compounds exhibit optical activity?\n1-methyl-4-(prop-1-en-2-yl)cyclohex-1-ene\n2,3,3,3-tetrafluoroprop-1-ene\ndi(cyclohex-2-en-1-ylidene)methane\n5-(5-methylhexan-2-ylidene)cyclopenta-1,3-diene\n3-(2-methylbut-1-en-1-ylidene)cyclohex-1-ene\n[1,1'-biphenyl]-3,3'-diol\n8,8-dichlorobicyclo[4.2.0]octan-7-one\ncyclopent-2-en-1-one\n\nA. 3\nB. 4\nC. 5\nD. 6\n\nPlease reason step-by-step and put your choice letter without any other text with \\boxed{} in the end.",
        "target": "The compounds 1-methyl-4-(prop-1-en-2-yl)cyclohex-1-ene, 3-(2-methylbut-1-en-1-ylidene)cyclohex-1-ene, di(cyclohex-2-en-1-ylidene)methane, and 8,8-dichlorobicyclo[4.2.0]octan-7-one are chiral molecules, and thus will be optically active. All the others have a mirror plane of symmetry, and will be achiral. Therefore, the answer is \\boxed{B}."
    },
    {
        "question": "A coating is applied to a substrate resulting in a perfectly smooth surface. The measured contact angles of this smooth coating are 132° and 102° for water and hexadecane respectively. The coating formulation is then modified and when now applied to the same type of substrate, a rough surface is produced. When a droplet of water or oil sits on the rough surface, the wettability of the surface can now be described by the Cassie-Baxter state. The water contact angle on the rough surface is now 148°. What would be the best estimate of the contact angle of a droplet of octane on the rough surface?\n\nA. 129°\nB. 134°\nC. 124°\nD. 139°\n\nPlease reason step-by-step and put your choice letter without any other text with \\boxed{} in the end.",
        "target": "In the Cassie-Baxter state, droplets are in contact with a non-uniform surface: some of the droplet is in contact with the coating and some with air. The contact angle (θCB) of a droplet in the Cassie-Baxter state is given by: cosθCB = f1.cosθ1 + f2.cosθ2, where f1 and f2 are the area fractions of the two components of the surface, in this case coating (f1) and air (f2). θ1 is the contact angle if the droplet was purely in contact with the coating, and θ2 is the contact angle if the droplet was purely in contact with air. First we need to calculate the f1 and f2 using the data we are given for water. We have θCB = 148°, θ1 = 132°, and θ2 is taken to be 180° (contact angle on air). We then have cos(148) = f1.cos(132) + f2.cos(180). By using f1 + f2 = 1, we can solve to give f1 = 0.46 and f2 = 0.54. Next we need to calculate the contact angle of hexadecane on the rough surface, we have θ1 = 102°, f1 = 0.46, f2 = 0.54, and θ2 is taken to be 180° (contact angle on air). Therefore, θCB = 129° for hexadecane. The question however asks about a droplet of octane. Octane is a shorter oil molecule than hexadecane and has a lower surface tension than hexadecane. For a given surface, the contact angle of octane is therefore always lower than for hexadecane. Therefore, the answer is \\boxed{C}."
    },
    {
        "question": "In a parallel universe where a magnet can have an isolated North or South pole, Maxwell's equations look different. But, specifically, which of those equations are different?\n\nA. The ones related to the circulation of the electric field and the divergence of the magnetic field.\nB. The ones related to the divergence and the curl of the magnetic field.\nC. The one related to the divergence of the magnetic field.\nD. The one related to the circulation of the magnetic field and the flux of the electric field.\n\nPlease reason step-by-step and put your choice letter without any other text with \\boxed{} in the end.",
        "target": "Let's call E and B the electric and magnetic fields, respectively: The ones related to the circulation of the electric field and the divergence of the magnetic field is correct, since knowing that magnets can have an isolated pole means that magnetic monopoles exist and, thus, the contributions of magnetic charges and magnetic currents must be included in the equations. The way to include them is to \"symmetry-copy\" the other equations, with the following dictionary: E <-> B; electric charge <-> magnetic charge; electric current <-> magnetic current. In this way, the equations that become modified, with added terms, are the ones related to the circulation (or curl, in differential form) of E, and to the divergence (or flux in integral form) of B. The ones related to the divergence and the curl of the magnetic field is incorrect, because the one with the curl does not change, since it already includes all symmetric contributions appearing in its symmetric equation (curl of electric field). The one related to the divergence of the magnetic field is incorrect because that equation does get changed, but it's not the only one; the equation for the curl (or circulation) of E also changes. The one related to the circulation of the magnetic field and the flux of the electric field is incorrect because none of those equations are changed, since they already include the symmetric terms appearing in their symmetric equations (circulation of E and flux of B). Therefore, the answer is \\boxed{A}."
    }
]

# Prompt construction
FEWSHOT_PROMPT = ""
for ex in FEWSHOT_EXAMPLES:
    FEWSHOT_PROMPT += f"Q: {ex['question']}\nA: {ex['target']}\n\n"

# Parser for extracting boxed answer (A/B/C/D)
def parse_answer(text):
    # 1. first try to find the answer in the text
    # support various formats: \boxed{A}, \\boxed{A}, $\boxed{A}$, \boxed {A} 等
    boxed_pattern = r"\\*boxed\s*[\{\[\(]?\s*([A-Da-d])\s*[\}\]\)]?"
    m = re.search(boxed_pattern, text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    
    # 2. secondary match "The answer is X" format
    # reference lm-evaluation-harness's strict-match method
    answer_pattern = r"(?:the\s+)?answer\s+is\s*:?\s*([A-Da-d])(?:\.|,|\s|$)"
    m = re.search(answer_pattern, text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    
    # 3. secondary match "(X)" format
    # reference lm-evaluation-harness's flexible-extract method
    # find the option in the parentheses at the end of the text
    paren_pattern = r"\(([A-Da-d])\)"
    matches = list(re.finditer(paren_pattern, text))
    if matches:
        # prioritize the last match (usually the final answer)
        return matches[-1].group(1).upper()
    
    # if no match, return empty string
    return ""

# sampling parameters
SAMPLING_PARAMS = dict(
    temperature=0.7,
    top_p=0.95,
    n=32,
    max_tokens=4096,
    # stop=["Q:", "</s>", "<|im_end|>", "\n\nQ:", "\n\nHuman:", "\n\nAssistant:", "Human:", "Assistant:"],
    seed=42,
)

def evaluate_gpqa_batch(gpqa_path, output_dir, model_path, idx_start, idx_end, n_sampling=32):
    """Evaluate a batch of GPQA questions for a single model"""
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading model: {model_path}")
    llm = LLM(model=model_path, dtype="auto", tensor_parallel_size=1)
    print("Model loaded successfully!")
    
    # Load data
    with open(gpqa_path, 'r') as f:
        all_data = [json.loads(line) for line in f]
    
    # Get batch
    data = all_data[idx_start:idx_end]
    
    print(f"Processing batch {idx_start}-{idx_end} ({len(data)} questions)")
    
    # Prepare prompts
    prompts = []
    golds = []
    questions = []
    for idx, item in enumerate(data):
        q = item["question"]
        gold = item["answer"].strip().upper()
        prompt = FEWSHOT_PROMPT + f"Q: {q}\nA: "
        # print the first prompt to check
        if idx == 0:
            print(prompt)
        prompts.append(prompt)
        golds.append(gold)
        questions.append(q)
    
    # Run inference
    params = SamplingParams(**{**SAMPLING_PARAMS, "n": n_sampling})
    outputs = llm.generate(prompts, params)
    
    # save all generations to JSONL
    jsonl_path = os.path.join(output_dir, f"gpqa_{idx_start}-{idx_end}.jsonl")
    with open(jsonl_path, 'w', encoding='utf-8') as jsonl_file:
        for idx, (q, gold, output) in enumerate(zip(questions, golds, outputs)):
            generations = [output.outputs[i].text for i in range(n_sampling)]
            json.dump({
                "question": q,
                "ground_truth": gold,
                "generations": generations
            }, jsonl_file, ensure_ascii=False)
            jsonl_file.write('\n')
    
    # Process results and save to CSV (simplified version)
    csv_path = os.path.join(output_dir, f"gpqa_{idx_start}-{idx_end}.csv")
    csv_rows = []
    
    for idx, (q, gold, output) in enumerate(zip(questions, golds, outputs)):
        generations = [output.outputs[i].text for i in range(n_sampling)]
        parsed = [parse_answer(gen) for gen in generations]
        em = [p == gold for p in parsed]
        
        row = {}
        for i in range(n_sampling):
            row[f"em_{i+1}"] = int(em[i])
        csv_rows.append(row)
    
    # Write CSV
    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = [f"em_{i+1}" for i in range(n_sampling)]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"Batch results saved to {csv_path} (CSV) and {jsonl_path} (JSONL)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpqa_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--idx_start", type=int, required=True)
    parser.add_argument("--idx_end", type=int, required=True)
    parser.add_argument("--n_sampling", type=int, default=32)
    
    args = parser.parse_args()
    
    evaluate_gpqa_batch(
        args.gpqa_path,
        args.output_dir,
        args.model_path,
        args.idx_start,
        args.idx_end,
        args.n_sampling
    ) 