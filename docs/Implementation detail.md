# 계층적 CFG 구현 규칙

코드(`src/hcfg/`)가 따르는 구현 규칙이다. 수식과 실험 결과는 최종 리포트(`docs/Hierarchical CFG - Separate Guidance to Avoid Forgetting Problem.md`)에 있다.

## 0. 용어 대응 (리포트 ↔ 연구 문서 ↔ 코드)

리포트는 가독성을 위해 일부 기호를 바꿔 썼다. 의미는 같고 이름만 다르다.

| 의미 | 최종 리포트 | `docs/diagnostics.md` 등 연구 문서 | 코드·설정·출력 파일 |
|---|---|---|---|
| 무조건부(unconditional) 조건 | $\varnothing$ | ∅, φ, Null, uncond | `null`, `uncond` (`model.uncond`, `SDXL.null_cond()`, 행 이름 `"null"`) |
| 파인튜닝이 ∅ 브랜치에 남긴 변화 ε(FT,∅) − ε(Base,∅) | $Diff_{\varnothing}$ | Diff_Null, D | `diff_null` (`ms_diff_null`, `rel_diff_null`, `cos_diff_null_g_domain`) |
| ε(FT,∅), ε(Base,∅) | $\varepsilon(\mathrm{FT}, \varnothing)$, $\varepsilon(\mathrm{Base}, \varnothing)$ | ε(FT,∅), ε(Base,∅) | `eps_ft_null`, `eps_base_null` |
| 도메인 가이던스 ε(FT,c) − ε(FT,∅) | $G_{domain}$ | G_domain, G_d | `g_domain` |
| 도메인 대안형 가이던스 ε(FT,c) − ε(Base,∅) | $G_{domain, hybrid}$ | G_domain,hybrid | `g_domain_hybrid` |
| 제어성 가이던스 ε(FT,c_refine) − ε(FT,c) | $G_{refine}$ | G_refine, G_r | `g_refine` |
| Base 모델의 제어성 가이던스 | $G_{refine}^{\mathrm{Base}}$ | G_refine^Base, G_r^B | `g_refine_base` |
| Diff_Null 분리 성분 (리포트 7장) | $Diff_{\parallel}$, $Diff_{\perp}$ | Diff_∥, Diff_⊥ | `par`, `perp` (`hcfg.split_par`, `hcfg.split_lambda`) |
| 분리 기준 벡터 ε(FT,c) − ε(Base,c) | $r$ | r, Diff_cond | `r` |
| 기본형 / 도메인 대안형 / 제어성 대안형 | 기본형 / 대안형(domain) / 대안형(refine) | 같음 | `hcfg.variant`: `basic` / `alt_domain` / `alt_refine` |
| ∅-anchoring 규제항 (미사용) | $\lambda \lVert \varepsilon(\mathrm{FT}, \varnothing) - \mathrm{sg}[\varepsilon(\mathrm{Base}, \varnothing)] \rVert^2$ | φ-anchoring | `train.phi_anchor_lambda` |
| refine 강도 / domain 강도 | $w_1$ / $w_2$ | w1 / w2 | `hcfg.w1` / `hcfg.w2`, 셀 이름 `w1_<a>_w2_<b>` |
| 육안 평가 점수 | D / C / K | 도메인 / 제어성 / 붕괴 | CSV 열 `domain` / `control` / `collapse` |

- `ms_*`는 원소당 평균 제곱(mean square), `cos_*`는 코사인 유사도다(7장).
- 이 프로젝트의 ∅는 zero 임베딩이다(`model.uncond: zeros`, 5장). 빈 문자열 `""`을 인코딩한 것(`empty`)과 다르다.

## 1. 용어

- domain: 데이터셋 스타일. CA-LoRA의 "concept"(스타일·시점)과 같은 대상이다.
- refine: 제어성(날씨·차종 등). c_refine = 제어 조건 프롬프트.
- concept / aug는 계층적 CFG 코드에서 쓰지 않는다.
  - concept는 CA-LoRA 선행연구 설명과 L_concept 표기에서만 쓴다.
  - aug(c_aug)는 CA 측정 단계(`hcfg.ca`) 전용이며, 계층적 CFG 코드에서 재사용하지 않는다.
- CA 측정 코드는 레퍼런스 레포(chs35/Improving-CA-LoRA)의 concept_loss 등을 재사용하지 않고 별도 모듈로 작성했다.

## 2. 기호

```
G_refine        = ε(FT, c_refine) − ε(FT, c)
G_domain        = ε(FT, c) − ε(FT, ∅)
Diff_Null       = ε(FT, ∅) − ε(Base, ∅)        # 진단값. guidance 축으로 쓰지 않음
G_domain,hybrid = ε(FT, c) − ε(Base, ∅) = G_domain + Diff_Null
G_refine^Base   = ε(Base, c_refine) − ε(Base, c)
```

## 3. 방법

- 기본형: ε(FT, ∅) + w1·G_refine + w2·G_domain
  - w1 = refine 강도, w2 = domain 강도 (순서 혼동 주의)
- 대안형(domain): ε(Base, ∅) 기준 + G_domain,hybrid
- 대안형(refine): G_refine 대신 G_refine^Base
- guidance 규약은 diffusers와 같다: ε_u + w(ε_c − ε_u), w = 1이면 가이던스 없음. (1 + w) 규약은 쓰지 않는다.
- guidance_rescale: 기본 off

## 4. 모델·어댑터 경로

- 기본형: 3 브랜치(∅, c, c_refine)가 모두 FT → LoRA merge 경로 허용
- 대안형·진단: Base forward가 필요 → merge 금지, adapter on/off(`disable_adapter` 또는 scale 0)로 전환
- 파인튜너별 Base:
  - vanilla LoRA / style CA-LoRA / CA-LoRA + condition dropout: adapter off = Base
  - DreamBooth (rank 64 LoRA, prior preservation): adapter off = Base, Diff_Null ≈ 0 (정확히 0은 아님)
  - Textual Inversion: UNet이 동일하고 ∅에 S*가 없음 → Diff_Null == 0 (정확히)

## 5. ∅(uncond) 정의

- SDXL `force_zeros_for_empty_prompt`: 0 텐서 ≠ `""` 인코딩. 하나로 통일하고 명시한다(`model.uncond`, 기본 `zeros`).
- CA-LoRA + condition dropout 대조군: 학습 시 드롭한 ∅와 추론 시 ∅가 반드시 같아야 한다.
- `added_cond_kwargs`(text_embeds(pooled), time_ids)를 브랜치 수만큼 정확히 concat한다. 암묵적 broadcast는 쓰지 않는다.

## 6. 프롬프트 (`configs/prompts.yaml`)

- c_refine은 c를 포함하는 상위 조건이어야 한다(c + 제어 조건).
- LoRA / CA-LoRA
  - c        = `Photorealistic first-person urban street view`
  - c_refine = `Photorealistic first-person urban street view, <날씨/차종>`
- DreamBooth / TI
  - c        = `Photorealistic first-person urban street view with sks style` (DreamBooth) / `... with S* style` (TI)
  - c_refine = c + `, <날씨/차종>`
  - TI의 `S*`는 학습된 placeholder 토큰(`<s*>`)으로 치환된다.

## 7. 진단 (Diff_Null, ‖G_refine‖² 학습 전후)

- 모든 브랜치는 한 루프 안에서 같은 z_t, 같은 t로 forward한다. 파이프라인을 브랜치별로 따로 돌려 차분하지 않는다.
- z_t 출처를 명시한다: q(z_t|x0) 전방 노이즈(held-out 이미지) vs 샘플링 궤적.
- t는 실제 timestep(0–999) 기준이다. scheduler step index를 쓰지 않는다.
- norm: 원소당 평균 제곱을 샘플 평균해 t별 곡선으로 저장한다.
- "학습 전" ‖G_refine‖² = ‖G_refine^Base‖² (중간 체크포인트가 아님)
- 재현성: 브랜치 간 generator 상태 공유, `torch.use_deterministic_algorithms` 점검, dtype 명시

## 8. 필수 테스트

| 항목 | 테스트 |
|---|---|
| w1 = w2 = w → c_refine 단일 프롬프트 표준 CFG와 같은 시드에서 latent 수치 일치 | `tests/test_hcfg_sampler.py::test_basic_equal_weights_is_standard_cfg` |
| Textual Inversion → Diff_Null == 0 | `tests/test_adapters_on_unet.py::test_textual_inversion_diff_null_is_zero`, `tests/test_diagnostics.py::test_textual_inversion_diff_null_exactly_zero` |
| LoRA scale 0 → ε(FT, ·) == ε(Base, ·) | `tests/test_adapters_on_unet.py::test_lora_scale_zero_equals_base`, `tests/test_diagnostics.py::test_scale_zero_gives_base` |

실제 SDXL(A100)에서의 같은 검사는 `scripts/gpu_check.py`의 `exact` 단계다(`docs/REPRODUCE.md`).
