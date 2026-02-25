#!/usr/bin/env bash
# 批量构建所有 FlashRAG 数据集
# 使用方法: bash scripts/data_process/build_all_datasets.sh

set -e

# 颜色输出
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "========================================"
echo "开始批量构建 FlashRAG 数据集"
echo "========================================"
echo ""

# 定义要处理的数据集列表（根据你的图片）
datasets=(
    "popqa"
    "hotpotqa"
    "2wikimultihopqa"
    "musique"
    "bamboogle"
    "triviaqa"# ARR的格式要求 New Official Review * denotes a required field Paper Summary* Please make sure that you are familiar with the latest version of ARR reviewer guidelines, especially with respect to AI assistance: https://aclrollingreview.org/reviewerguidelines#-task-3-write-a-strong-review Note that the reviewer names are anonymous to the authors, but are VISIBLE to the senior researchers serving as area chairs, senior chairs and program chairs. Authors will have an opportunity to submit issue reports for problematic reviews, to be considered by area chairs (https://aclrollingreview.org/authors#step2.2). Highly problematic reviews may result in penalties, and great reviews may result in awards (https://aclrollingreview.org/incentives2025) Describe what this paper is about. This should help the program and area chairs to understand the topic of the work and highlight any possible misunderstandings. Maximum length 20000 characters. Write Preview TeX is supported Summary Of Strengths* What are the major reasons to publish this paper at a selective *ACL venue? These could include novel and useful methodology, insightful empirical results or theoretical analysis, clear organization of related literature, or any other reason why interested readers of *ACL papers may find the paper useful. Maximum length 20000 characters. Write Preview TeX is supported Summary Of Weaknesses* What are the concerns that you have about the paper that would cause you to favor prioritizing other high-quality papers that are also under consideration for publication? These could include concerns about correctness of the results or argumentation, limited perceived impact of the methods or findings (note that impact can be significant both in broad or in narrow sub-fields), lack of clarity in exposition, or any other reason why interested readers of *ACL papers may gain less from this paper than they would from other papers under consideration. Where possible, please number your concerns so authors may respond to them individually. Maximum length 20000 characters. If the paper is a resubmission, please discuss whether previous feedback has been adequately addressed (revision notes should be in the submission under 'explanation of revisions PDF'). Write Preview TeX is supported Comments Suggestions And Typos* If you have any comments to the authors about how they may improve their paper, other than addressing the concerns above, please list them here. Maximum length 20000 characters. Write Preview TeX is supported Confidence* 5 = Positive that my evaluation is correct. I read the paper very carefully and am familiar with related work. 4 = Quite sure. I tried to check the important points carefully. It's unlikely, though conceivable, that I missed something that should affect my ratings. 3 = Pretty sure, but there's a chance I missed something. Although I have a good feel for this area in general, I did not carefully check the paper's details, e.g., the math or experimental design. 2 = Willing to defend my evaluation, but it is fairly likely that I missed some details, didn't understand some central points, or can't be sure about the novelty of the work. 1 = Not my area, or paper is very hard to understand. My evaluation is just an educated guess. Soundness* Given that this is a short/long paper, is it sufficiently sound and thorough? Does it clearly state scientific claims and provide adequate support for them? For experimental papers: consider the depth and/or breadth of the research questions investigated, technical soundness of experiments, methodological validity of evaluation. For position papers, surveys: consider whether the current state of the field is adequately represented and main counter-arguments acknowledged. For resource papers: consider the data collection methodology, resulting data & the difference from existing resources are described in sufficient detail. 5 = Excellent: This study is one of the most thorough I have seen, given its type. 4.5 4 = Strong: This study provides sufficient support for all of its claims. Some extra experiments could be nice, but not essential. 3.5 3 = Acceptable: This study provides sufficient support for its main claims. Some minor points may need extra support or details. 2.5 2 = Poor: Some of the main claims are not sufficiently supported. There are major technical/methodological problems. 1.5 1 = Major Issues: This study is not yet sufficiently thorough to warrant publication or is not relevant to ACL. Excitement* How exciting is this paper for you? Excitement is SUBJECTIVE, and does not necessarily follow what is popular in the field. We may perceive papers as transformational/innovative/surprising, e.g. because they present conceptual breakthroughs or evidence challenging common assumptions/methods/datasets/metrics. We may be excited about the possible impact of the paper on some community (not necessarily large or our own), e.g. lowering barriers, reducing costs, enabling new applications. We may be excited for papers that are relevant, inspiring, or useful for our own research. These factors may combine in different ways for different reviewers. 5 = Highly Exciting: I would recommend this paper to others and/or attend its presentation in a conference. 4.5 4 = Exciting: I would mention this paper to others and/or make an effort to attend its presentation in a conference. 3.5 3 = Interesting: I might mention some points of this paper to others and/or attend its presentation in a conference if there's time. 2.5 2 = Potentially Interesting: this paper does not resonate with me, but it might with others in the *ACL community. 1.5 1 = Not Exciting: this paper does not resonate with me, and I don't think it would with others in the *ACL community (e.g. it is in no way related to computational processing of language). Overall Assessment* If this paper was committed to an *ACL conference, do you believe it should be accepted? If you recommend conference, Findings and or even award consideration, you can still suggest minor revisions (e.g. typos, non-core missing refs, etc.). Outstanding papers should be either fascinating, controversial, surprising, impressive, or potentially field-changing. Awards will be decided based on the camera-ready version of the paper. ACL award policy: https://www.aclweb.org/adminwiki/index.php/ACL_Conference_Awards_Policy Main vs Findings papers: the main criteria for Findings are soundness and reproducibility. Conference recommendations may also consider novelty, impact and other factors. 5 = Consider for Award: I think this paper could be considered for an outstanding paper award at an *ACL conference (up to top 2.5% papers). 4.5 = Borderline Award 4 = Conference: I think this paper could be accepted to an *ACL conference. 3.5 = Borderline Conference 3 = Findings: I think this paper could be accepted to the Findings of the ACL. 2.5 = Borderline Findings 2 = Resubmit next cycle: I think this paper needs substantial revisions that can be completed by the next ARR cycle. 1.5 = Resubmit after next cycle: I think this paper needs substantial revisions that cannot be completed by the next ARR cycle. 1 = Do not resubmit: this paper has to be fully redone, or it is not relevant to the *ACL community (e.g. it is in no way related to computational processing of language).
)

# 记录成功和失败的数据集
success_datasets=()
failed_datasets=()

# 处理每个数据集
for dataset in "${datasets[@]}"; do
    echo ""
    echo "----------------------------------------"
    echo -e "${YELLOW}[处理中] 数据集: $dataset${NC}"
    echo "----------------------------------------"
    
    if python scripts/data_process/build_search_dataset.py --dataset_name "$dataset"; then
        echo -e "${GREEN}✓ $dataset 构建成功${NC}"
        success_datasets+=("$dataset")
    else
        echo -e "${RED}✗ $dataset 构建失败（继续下一个）${NC}"
        failed_datasets+=("$dataset")
    fi
done

# 输出汇总报告
echo ""
echo "========================================"
echo "批量构建完成！"
echo "========================================"
echo ""

if [ ${#success_datasets[@]} -gt 0 ]; then
    echo -e "${GREEN}✓ 成功构建的数据集 (${#success_datasets[@]}/${#datasets[@]}):${NC}"
    for dataset in "${success_datasets[@]}"; do
        echo "  - $dataset → ./data/${dataset}_search/"
    done
fi

if [ ${#failed_datasets[@]} -gt 0 ]; then
    echo ""
    echo -e "${RED}✗ 失败的数据集 (${#failed_datasets[@]}/${#datasets[@]}):${NC}"
    for dataset in "${failed_datasets[@]}"; do
        echo "  - $dataset"
    done
    echo ""
    echo "重试失败的数据集："
    for dataset in "${failed_datasets[@]}"; do
        echo "  python scripts/data_process/build_search_dataset.py --dataset_name $dataset"
    done
    exit 1
fi

echo ""
echo "全部数据集构建完成！🎉"

