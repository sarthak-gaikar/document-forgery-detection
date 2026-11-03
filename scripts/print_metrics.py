import csv
rows = list(csv.DictReader(open('outputs/logs/dual_branch_forgery_net/metrics.csv')))
val_rows = [r for r in rows if r['split'] == 'val']
print('Epoch | Val Dice | Val IoU  | Val Loss | Cls Acc')
print('------|----------|----------|----------|--------')
for r in val_rows:
    ep = r['epoch']
    dice = float(r['dice'])
    iou = float(r['iou'])
    loss = float(r['loss'])
    cls_acc = float(r['cls_acc'])
    print(f"  {ep:>2}  | {dice:.4f}   | {iou:.4f}  | {loss:.4f}   | {cls_acc:.3f}")

