import json
from pathlib import Path
from ultralytics import YOLO


def collect_metrics(metrics, model):
    per_class = []
    for i, c in enumerate(metrics.box.ap_class_index):
        per_class.append({
            "class":    model.names[c],
            "mAP50":    round(float(metrics.box.ap50[i]), 4),
            "mAP50-95": round(float(metrics.box.ap[i]),   4),
            "precision":round(float(metrics.box.p[i]), 4),
            "recall":   round(float(metrics.box.r[i]), 4),
        })

    return {
        "mAP50":     round(float(metrics.box.map50), 4),
        "mAP50-95":  round(float(metrics.box.map),   4),
        "precision": round(float(metrics.box.mp), 4),
        "recall":    round(float(metrics.box.mr), 4),
        "per_class": per_class,
    }


def print_metrics(data, split_label):
    print("\n" + "="*50)
    print(f"MÉTRICAS GERAIS [{split_label}]:")
    print("="*50)
    print(f"mAP50:95  : {data['mAP50-95']:.4f}")
    print(f"mAP50     : {data['mAP50']:.4f}")
    print(f"Precision : {data['precision']:.4f}")
    print(f"Recall    : {data['recall']:.4f}")

    print("\n" + "="*65)
    print(f"MÉTRICAS POR CLASSE [{split_label}]:")
    print("="*65)
    print(f"{'Classe':<20} | {'mAP50':<7} | {'mAP50-95':<8} | {'Prec.':<7} | {'Recall':<7}")
    print("-" * 57)
    for c in data["per_class"]:
        print(f"{c['class'][:19]:<20} | {c['mAP50']:.4f}  | {c['mAP50-95']:.4f}   | {c['precision']:.4f}  | {c['recall']:.4f}")
    print("="*65 + "\n")


if __name__ == '__main__':
    model = YOLO("modelos/yolov8n/weights/best.pt")

    metrics_val  = model.val(data="IA-8230365-8230196-13/data.yaml", split="val")
    metrics_test = model.val(data="IA-8230365-8230196-13/data.yaml", split="test")

    val_data  = collect_metrics(metrics_val,  model)
    test_data = collect_metrics(metrics_test, model)

    print_metrics(val_data,  "VAL")
    print_metrics(test_data, "TEST")

    out_path = Path("modelos/yolov8n/eval.json")
    model_name = out_path.parent.name
    output = {"model": model_name, "val": val_data, "test": test_data}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Métricas guardadas em {out_path}")
