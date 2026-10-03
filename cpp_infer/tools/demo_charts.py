#!/usr/bin/env python3
"""Render bilingual demo charts from checked-in Runtime evidence (no new runs).

Usage: python cpp_infer/tools/demo_charts.py [--check]
Requires matplotlib. Chinese labels use Microsoft YaHei, Noto Sans CJK SC,
or another installed CJK font; SVG glyphs are embedded for offline portability.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import MaxNLocator


ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "docs/assets/engineering"
EVIDENCE = ROOT / "docs/demo/evidence.json"
QUANT_COMPARE = "cpp_infer/results/s2_01/round2/benchmark/comparison_u8s8.json"
QUANT_BENCH = "cpp_infer/results/s2_01/round2/benchmark/fp32_cpu_release.json"
QUANT_QUALITY = "cpp_infer/results/s2_01/round2/correctness_u8s8.json"
WINDOWS_COMPARE = "cpp_infer/results/s2_03/windows_x86_64/comparison.json"
LINUX_COMPARE = "cpp_infer/results/s2_03/linux_x86_64/performance/batch_comparison.json"
INTEGRATION = "cpp_infer/results/s2_03/int8_integration/verification_summary.json"
STONE, CLAY, INK, MUTED, GRID, AMBER = (
    "#5c564d", "#c96442", "#1f1e1c", "#6b675e", "#ebe6db", "#a8770f"
)


def read(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8-sig"))


def bilingual(zh, en):
    return {"zh": zh, "en": en}


def source(path, zh, en):
    return {"path": path, "label": bilingual(zh, en)}


def metric(key, zh, en, unit, fp32, int8, path, pointer, environment):
    return {
        "id": key, "label": bilingual(zh, en), "unit": unit,
        "fp32": fp32, "int8": int8, "source": path,
        "json_pointer": pointer, "environment_id": environment,
    }


def collect_evidence():
    comparison, quality = map(read, (QUANT_COMPARE, QUANT_QUALITY))
    task_quality = quality["task_quality"]
    size = comparison["models"]
    latency = comparison["latency_ms"]["pipeline"]["mean"]
    drop_pp = -task_quality["deltas"]["map50_delta"] * 100
    benchmark_env = bilingual(
        "Windows x86_64 · C++ Release · ORT 1.19.2 CPU · 单会话线程 1/1 · 固定单图，预热 10 次、测量 100 次（2026-08-27）",
        "Windows x86_64 · C++ Release · ORT 1.19.2 CPU · session threads 1/1 · one fixed image, 10 warmups / 100 repeats (2026-08-27)",
    )
    quality_env = bilingual(
        "361 张验证图 · 857 个标注框 · score floor 0.001 · class-agnostic NMS 0.45 · COCO 101 点 AP",
        "361 validation images · 857 ground-truth boxes · score floor 0.001 · class-agnostic NMS 0.45 · COCO 101-point AP",
    )
    metrics = [
        metric("model_size", "模型文件", "Model file", "MB (10^6 bytes)",
               size["fp32"]["file_size_bytes"] / 1e6,
               size["int8"]["file_size_bytes"] / 1e6,
               QUANT_COMPARE, "/models", "quantization_benchmark"),
        metric("pipeline_mean", "Pipeline 平均延迟", "Mean pipeline latency", "ms",
               latency["fp32"], latency["int8"], QUANT_COMPARE,
               "/latency_ms/pipeline/mean", "quantization_benchmark"),
        metric("map50", "验证集 mAP50", "Validation mAP50", "%",
               task_quality["fp32"]["map50"] * 100, task_quality["int8"]["map50"] * 100,
               QUANT_QUALITY, "/task_quality", "quantization_quality"),
        metric("map50_95", "验证集 mAP50–95", "Validation mAP50–95", "%",
               task_quality["fp32"]["map50_95"] * 100,
               task_quality["int8"]["map50_95"] * 100,
               QUANT_QUALITY, "/task_quality", "quantization_quality"),
    ]
    ious = quality["protocol"]["iou_thresholds"]
    curve = {"unit": "%", "iou_thresholds": ious, "environment_id": "quantization_quality",
             "source": QUANT_QUALITY, "json_pointer": "/task_quality/{fp32,int8}/per_class/*/ap_by_iou"}
    for precision in ("fp32", "int8"):
        classes = task_quality[precision]["per_class"]
        curve[precision] = [sum(c["ap_by_iou"][f"{iou:.2f}"] for c in classes) / len(classes) * 100 for iou in ious]

    quantization = {
        "title": bilingual("量化：看见收益，也保留质量结论", "Quantization: measure the gains and the quality trade-off"),
        "environment": benchmark_env, "metrics": metrics, "quality_curve": curve,
        "conclusion": bilingual(
            f"正式 U8S8 模型缩小 {size['size']['reduction_fraction'] * 100:.1f}%，单图 Pipeline 平均延迟由 {latency['fp32']:.2f} ms 降至 {latency['int8']:.2f} ms。下方质量曲线来自独立的 361 图验证集。",
            f"The formal U8S8 model is {size['size']['reduction_fraction'] * 100:.1f}% smaller; mean single-image pipeline latency falls from {latency['fp32']:.2f} to {latency['int8']:.2f} ms. The quality curve uses a separate 361-image validation set.",
        ),
        "quality": {
            "passed": quality["passed"], "unit": "percentage points",
            "environment_id": "quantization_quality", "source": QUANT_QUALITY,
            "map50_drop_percentage_points": drop_pp,
            "gate_percentage_points": task_quality["gates"]["map50_absolute_drop_max"] * 100,
            "agreement_precision": quality["product_detection_difference"]["metrics"]["int8_agreement_precision"],
            "agreement_precision_gate": quality["product_detection_difference"]["gates"]["int8_agreement_precision_min"],
            "note": bilingual(
                f"严格质量门禁未通过：mAP50 下降 {drop_pp:.3f} 个百分点，略高于 1.000 的上限；30 图产品一致性中的 INT8 agreement precision 和置信度误差 P95 也未过门禁。该结果用于展示工程取舍，未改写为无损量化。",
                f"Strict quality gates remain unmet: mAP50 drops {drop_pp:.3f} percentage points, above the 1.000-point limit. INT8 agreement precision and confidence-error P95 also miss their gates on the 30-image product comparison. This is a measured trade-off, not lossless quantization.",
            ),
        },
        "chart": {lang: f"../assets/engineering/quantization.{lang}.svg" for lang in ("zh", "en")},
        "sources": [source(QUANT_COMPARE, "同协议 FP32 / U8S8 benchmark", "Matched FP32 / U8S8 benchmark"),
                    source(QUANT_QUALITY, "361 图质量与 30 图产品一致性", "361-image quality and 30-image product consistency"),
                    source(QUANT_BENCH, "环境、线程与测量口径", "Environment, threading and measurement protocol")],
    }
    platforms = []
    for platform_id, path, summary_path, label, context in [
        ("windows", WINDOWS_COMPARE, "cpp_infer/results/s2_03/windows_x86_64/workers_4/batch_summary.json",
         bilingual("Windows x86_64", "Windows x86_64"),
         bilingual("原生 Windows · MSVC 19.50 · OpenCV 4.8.0", "Native Windows · MSVC 19.50 · OpenCV 4.8.0")),
        ("linux", LINUX_COMPARE, "cpp_infer/results/s2_03/linux_x86_64/performance/batch_workers_4/batch_summary.json",
         bilingual("WSL2 / Linux x86_64", "WSL2 / Linux x86_64"),
         bilingual("WSL2 Ubuntu 24.04 · GCC 13.3 · OpenCV 4.6.0 · ext4", "WSL2 Ubuntu 24.04 · GCC 13.3 · OpenCV 4.6.0 · ext4")),
    ]:
        data, summary = read(path), read(summary_path)
        memory = data["peak_process_memory_bytes"]
        platforms.append({
            "id": platform_id, "label": label, "environment": context,
            "source": path, "runtime_source": summary_path,
            "throughput_images_per_second": {key: data["throughput_images_per_second"][key] for key in ("workers_1", "workers_4")},
            "throughput_unit": "images/s", "throughput_ratio": data["throughput_images_per_second"]["workers_4_div_workers_1"],
            "peak_process_memory_mib": {key: memory[key] / 2 ** 20 for key in ("workers_1", "workers_4")},
            "memory_unit": "MiB", "memory_metric": memory["metric"],
            "memory_scope": "process-lifetime high-water mark",
            "images": data["comparability"]["compared_item_count"], "images_unit": "images",
            "queue_capacity": data["comparability"]["queue_capacity"], "queue_unit": "tasks",
            "queue_peak_depth_workers_4": summary["queue"]["peak_depth"],
            "producer_wait_count_workers_4": summary["queue"]["producer_wait_count"],
            "json_byte_equal": data["comparability"]["detection_json_byte_equal"],
        })
    batch = {
        "title": bilingual("有界并发：吞吐量与内存一起看", "Bounded concurrency: throughput alongside memory"),
        "environment": bilingual(
            "FP32 · 361 图 · queue capacity 8 · JSON 输出 · 每 worker 独立 Pipeline / Session · ORT 线程 1/1（2026-08-30）",
            "FP32 · 361 images · queue capacity 8 · JSON output · independent Pipeline / Session per worker · ORT threads 1/1 (2026-08-30)",
        ),
        "platforms": platforms,
        "conclusion": bilingual(
            f"worker 从 1 增至 4 后，Windows 吞吐量为 {platforms[0]['throughput_ratio']:.2f} 倍，WSL2/Linux 为 {platforms[1]['throughput_ratio']:.2f} 倍；进程峰值内存随独立会话增加。两组各自的 361 份逐图 JSON 均字节一致，队列峰值均未超过容量 8。",
            f"Moving from 1 to 4 workers yields {platforms[0]['throughput_ratio']:.2f}× throughput on Windows and {platforms[1]['throughput_ratio']:.2f}× on WSL2/Linux, with higher process peak memory. Within each comparison, all 361 per-image JSON files are byte-identical and peak queue depth stays within capacity 8.",
        ),
        "note": bilingual(
            "仅在各自记录环境内比较。Windows Peak Working Set 与 Linux peak RSS 是不同口径；这是图像级 batch=1 并发，不是张量 batching。",
            "Compare runs only within each recorded environment. Windows Peak Working Set and Linux peak RSS are distinct metrics. This is concurrent image-level batch=1, not tensor batching.",
        ),
        "chart": {lang: f"../assets/engineering/batch-throughput.{lang}.svg" for lang in ("zh", "en")},
        "sources": [source(WINDOWS_COMPARE, "Windows 361 图并发对比", "Windows 361-image worker comparison"),
                    source(LINUX_COMPARE, "WSL2/Linux 361 图并发对比", "WSL2/Linux 361-image worker comparison"),
                    source(platforms[0]["runtime_source"], "Windows 队列与运行环境", "Windows queue and runtime environment"),
                    source(platforms[1]["runtime_source"], "WSL2/Linux 队列与运行环境", "WSL2/Linux queue and runtime environment")],
    }
    integration = read(INTEGRATION)
    assert integration["passed"] and not integration["linux_aarch64_int8_qemu"]["performance_or_memory_publishable"]
    portability = {
        "title": bilingual("同一条 Runtime 主链，三个验证环境", "One Runtime pipeline across three validation environments"),
        "conclusion": bilingual(
            "FP32 与正式 U8S8 复用 RuntimeConfig → DetectorPipeline → BatchRunner；AArch64 通过交叉编译和 QEMU 用户态验证构建、加载及功能。",
            "FP32 and formal U8S8 reuse RuntimeConfig → DetectorPipeline → BatchRunner. AArch64 cross-build and QEMU user-mode runs validate build, loading and functional portability.",
        ),
        "rows": [
            {"id": "windows", "name": "Windows x86_64", "environment": platforms[0]["environment"],
             "capabilities": {"single_image": True, "directory_manifest": True, "fp32_int8": True, "bounded_queue": True},
             "note": bilingual("原生 CPU 验证；Qt 客户端的当前交付平台。", "Native CPU validation; current delivery platform for the Qt client.")},
            {"id": "linux", "name": "WSL2 / Linux x86_64", "environment": platforms[1]["environment"],
             "capabilities": {"single_image": True, "directory_manifest": True, "fp32_int8": True, "bounded_queue": True},
             "note": bilingual("Linux Runtime / CLI 验证，含目录与 manifest 一致性。", "Linux Runtime / CLI validation, including directory / manifest parity.")},
            {"id": "aarch64", "name": "Linux AArch64 / QEMU",
             "environment": bilingual("x86_64 主机交叉编译 · GCC 13 · QEMU 用户态", "Cross-compiled on x86_64 · GCC 13 · QEMU user mode"),
             "capabilities": {"single_image": True, "directory_manifest": True, "fp32_int8": True, "bounded_queue": True},
             "note": bilingual("仅证明功能可移植；不提供 ARM 实机速度、内存或功耗结论。", "Functional portability only; no native ARM latency, memory or power claim.")},
        ],
        "sources": [source(INTEGRATION, "FP32 / U8S8 跨平台集成记录", "FP32 / U8S8 cross-platform integration record"),
                    source("cpp_infer/results/s2_03/linux_aarch64_qemu/verification.md", "AArch64 构建、加载与功能验证", "AArch64 build, loader and functional validation")],
    }
    return {
        "schema_version": 1, "generator": "cpp_infer/tools/demo_charts.py",
        "environments": {"quantization_benchmark": benchmark_env, "quantization_quality": quality_env},
        "quantization": quantization, "batch": batch, "platforms": portability,
    }


def style(lang):
    fonts = {font.name for font in font_manager.fontManager.ttflist}
    cjk = next((name for name in ("Microsoft YaHei", "Noto Sans CJK SC", "Source Han Sans SC", "SimHei") if name in fonts), None)
    if lang == "zh" and cjk is None:
        raise RuntimeError("Chinese charts need Microsoft YaHei or Noto Sans CJK SC installed.")
    plt.rcParams.update({
        "font.family": [cjk, "DejaVu Sans"] if lang == "zh" else ["DejaVu Sans"],
        "font.size": 11, "axes.titlesize": 13, "axes.titleweight": "bold",
        "axes.titlecolor": INK, "axes.labelcolor": MUTED, "text.color": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": GRID,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.spines.left": False, "axes.spines.bottom": False,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "svg.fonttype": "path", "svg.hashsalt": "yolo-defect-demo-v1",
    })


def horizontal_comparison(ax, values, title, unit, labels, fmt=".2f"):
    ax.barh([1, 0], values, height=0.48, color=[STONE, CLAY], zorder=3)
    ax.set_yticks([1, 0], labels)
    ax.tick_params(axis="both", length=0, pad=7)
    ax.set_xlim(0, max(values) * 1.28)
    ax.set_ylim(-0.6, 1.6)
    ax.set_title(title, loc="left", pad=17)
    ax.set_xlabel(unit, loc="right", labelpad=8)
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.xaxis.grid(True, color=GRID, zorder=0)
    for y, value in zip([1, 0], values):
        ax.text(value + max(values) * 0.035, y, format(value, fmt), va="center", fontweight="bold", color=INK)


def save_figure(fig, path):
    fig.savefig(path, format="svg", metadata={"Date": None, "Creator": "demo_charts.py / matplotlib"})
    plt.close(fig)


def quantization_chart(data, lang):
    style(lang)
    quant = data["quantization"]
    fig = plt.figure(figsize=(12, 7.8))
    grid = fig.add_gridspec(2, 2, height_ratios=[1, 1.6], left=0.1, right=0.96, top=0.81, bottom=0.16, wspace=0.43, hspace=0.7)
    fig.text(0.05, 0.95, "FP32 → INT8 / U8S8", fontsize=19, fontweight="bold")
    fig.text(0.05, 0.91, bilingual("同协议性能测量 + 独立验证集质量评估", "Matched performance protocol + independent validation-set quality")[lang], color=MUTED, fontsize=11)
    for index in range(2):
        item = quant["metrics"][index]
        horizontal_comparison(fig.add_subplot(grid[0, index]), [item["fp32"], item["int8"]], item["label"][lang], "MB" if index == 0 else "ms", ["FP32", "INT8"])
    curve = quant["quality_curve"]
    ax = fig.add_subplot(grid[1, :])
    for key, color, label in [("fp32", STONE, "FP32"), ("int8", CLAY, "INT8 / U8S8")]:
        ax.plot(curve["iou_thresholds"], curve[key], "o-", color=color, label=label, linewidth=2.4, markersize=5)
    ax.set_ylim(0, 80)
    ax.set_xlim(0.485, 0.965)
    ax.set_xticks(curve["iou_thresholds"])
    ax.set_yticks([0, 20, 40, 60, 80])
    ax.grid(axis="y", color=GRID)
    ax.tick_params(length=0, pad=6)
    ax.set_xlabel(bilingual("匹配 IoU 阈值", "Matching IoU threshold")[lang], labelpad=8)
    ax.set_ylabel("mAP (%)", labelpad=12)
    ax.set_title(bilingual("361 图验证集：定位要求提高时的检测质量", "361-image validation: detection quality at stricter localization thresholds")[lang], loc="left", pad=15)
    ax.legend(frameon=False, loc="upper right", ncols=2)
    quality_values = quant["metrics"][2]
    quality_label = f"mAP50  {quality_values['fp32']:.2f}% → {quality_values['int8']:.2f}%"
    quality_drop = quant["quality"]["map50_drop_percentage_points"]
    fig.text(0.05, 0.072, quality_label + bilingual(f"   ·   下降 {quality_drop:.3f} 个百分点", f"   ·   {quality_drop:.3f}-point drop")[lang], color=AMBER, fontsize=11)
    fig.text(0.05, 0.03, bilingual("性能：Windows x86_64 / CPU / 预热 10 + 测量 100   ·   来源：s2_01/round2", "Performance: Windows x86_64 / CPU / 10 warmups + 100 repeats   ·   Source: s2_01/round2")[lang], color=MUTED, fontsize=9.5)
    save_figure(fig, ASSETS / f"quantization.{lang}.svg")


def batch_chart(data, lang):
    style(lang)
    fig = plt.figure(figsize=(12, 7.0))
    grid = fig.add_gridspec(2, 2, left=0.1, right=0.96, top=0.82, bottom=0.19, wspace=0.43, hspace=0.65)
    fig.text(0.05, 0.95, bilingual("并发收益与内存成本", "Concurrency gains and memory cost")[lang], fontsize=19, fontweight="bold")
    fig.text(0.05, 0.90, bilingual("361 图 · FP32 · 有界队列容量 8 · 逐图 JSON 字节一致", "361 images · FP32 · bounded queue capacity 8 · byte-identical per-image JSON")[lang], color=MUTED, fontsize=11)
    labels = ["1 worker", "4 workers"]
    for column, platform in enumerate(data["batch"]["platforms"]):
        throughput = platform["throughput_images_per_second"]
        name = platform["label"][lang]
        horizontal_comparison(fig.add_subplot(grid[0, column]), [throughput["workers_1"], throughput["workers_4"]], f"{name}   /   {platform['throughput_ratio']:.2f}×", "images/s", labels)
        memory = platform["peak_process_memory_mib"]
        title = "Peak Working Set" if platform["memory_metric"] == "peak_working_set" else "Peak RSS"
        horizontal_comparison(fig.add_subplot(grid[1, column]), [memory["workers_1"], memory["workers_4"]], title, "MiB", labels, ".1f")
    fig.text(0.05, 0.09, bilingual("每个 worker 独立持有 Pipeline / ORT Session；吞吐量增加，进程峰值内存也增加。", "Each worker owns a Pipeline / ORT Session: throughput grows alongside process peak memory.")[lang], color=INK, fontsize=10.5)
    fig.text(0.05, 0.035, bilingual("各平台仅内部比较；两种内存指标不可直接对比。Linux 运行于 WSL2 / ext4。来源：s2_03", "Within-platform comparisons only; memory metrics are not interchangeable. Linux runs on WSL2 / ext4. Source: s2_03")[lang], color=MUTED, fontsize=9)
    save_figure(fig, ASSETS / f"batch-throughput.{lang}.svg")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Check evidence JSON against recorded sources without writing files")
    args = parser.parse_args()
    data = collect_evidence()
    encoded = json.dumps(data, ensure_ascii=False, indent=2) + "\n"
    if args.check:
        if not EVIDENCE.exists() or EVIDENCE.read_text(encoding="utf-8") != encoded:
            raise SystemExit("Demo evidence is stale; run demo_charts.py to regenerate it.")
        for section in ("quantization", "batch"):
            for path in data[section]["chart"].values():
                if not (EVIDENCE.parent / path).is_file():
                    raise SystemExit(f"Missing chart: {path}")
        print("Demo evidence matches recorded sources; four bilingual charts are present.")
        return
    ASSETS.mkdir(parents=True, exist_ok=True)
    with EVIDENCE.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(encoded)
    for lang in ("zh", "en"):
        quantization_chart(data, lang)
        batch_chart(data, lang)
    print(f"Wrote {EVIDENCE.relative_to(ROOT)} and four bilingual SVG charts.")


if __name__ == "__main__":
    main()
