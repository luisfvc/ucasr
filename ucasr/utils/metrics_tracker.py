import time

from ucasr.utils.colored_printing import BColors

def initialize_metrics_summary():
    """ initialize empty dictionary to track evaluation metrics during training """

    empty_summary = dict(
        epoch=[],

        # training metrics
        tr_loss=[],
        tr_mean_dist=[],
        tr_mean_rank=[],
        tr_med_rank=[],
        tr_map=[],

        # validation metrics
        va_loss=[],
        va_mean_dist=[],
        va_mean_rank=[],
        va_med_rank=[],
        va_map=[],
    )

    return empty_summary


def update_metrics_summary(summary_input, epoch, tr_metrics, va_metrics):
    """ update metrics after epoch iteration on tr/val sets """

    metrics_summary = summary_input.copy()

    labels = ['loss', 'mean_dist', 'mean_rank', 'med_rank', 'map']

    # updating epoch counter
    metrics_summary['epoch'].append(epoch)

    # updating metrics
    for i, metrics in zip(['tr_', 'va_'], [tr_metrics, va_metrics]):
        for lab in labels:
            metrics_summary[i + lab].append(metrics[lab])

    return metrics_summary


def print_epoch_stats(metrics, epoch_now, patience):
    """ print metrics of the current epoch """

    first = metrics['epoch'][-1] == 0

    best_tr_loss = 1e7 if first else min(metrics['tr_loss'][:-1])
    best_tr_dist = 1e7 if first else min(metrics['tr_mean_dist'][:-1])
    best_tr_medr = 1e7 if first else min(metrics['tr_med_rank'][:-1])
    best_tr_map = 0 if first else max(metrics['tr_map'][:-1])

    best_va_loss = 1e7 if first else min(metrics['va_loss'][:-1])
    best_va_dist = 1e7 if first else min(metrics['va_mean_dist'][:-1])
    best_va_medr = 1e7 if first else min(metrics['va_med_rank'][:-1])
    best_va_map = 0 if first else max(metrics['va_map'][:-1])

    tr_loss, tr_dist = metrics['tr_loss'][-1], metrics['tr_mean_dist'][-1]
    tr_medr, tr_map = metrics['tr_med_rank'][-1], metrics['tr_map'][-1]

    va_loss, va_dist = metrics['va_loss'][-1], metrics['va_mean_dist'][-1]
    va_medr, va_map = metrics['va_med_rank'][-1], metrics['va_map'][-1]

    # colored printer
    col = BColors()

    # change color if metric is improved
    txt_tr_loss = 'tr_loss: %.5f, ' % tr_loss
    if tr_loss < best_tr_loss:
        txt_tr_loss = col.colored(txt_tr_loss, BColors.OKGREEN)

    txt_va_loss = 'va_loss: %.5f, ' % va_loss
    if va_loss < best_va_loss:
        txt_va_loss = col.colored(txt_va_loss, BColors.OKGREEN)

    txt_tr_dist = 'tr_dist: %.5f, ' % tr_dist
    if tr_dist < best_tr_dist:
        txt_tr_dist = col.colored(txt_tr_dist, BColors.OKGREEN)

    txt_va_dist = 'va_dist: %.5f, ' % va_dist
    if va_dist < best_va_dist:
        txt_va_dist = col.colored(txt_va_dist, BColors.OKGREEN)

    txt_tr_map = 'tr_map: %.2f, ' % (100 * tr_map)
    if tr_map > best_tr_map:
        txt_tr_map = col.colored(txt_tr_map, BColors.OKGREEN)

    txt_va_map = 'va_map: %.2f, ' % (100 * va_map)
    if va_map > best_va_map:
        txt_va_map = col.colored(txt_va_map, BColors.OKGREEN)

    txt_tr_med_rank = 'tr_medr: %.2f, ' % tr_medr
    if tr_medr < best_tr_medr:
        txt_tr_med_rank = col.colored(txt_tr_med_rank, BColors.OKGREEN)

    txt_va_med_rank = 'va_medr: %.2f, ' % va_medr
    if va_medr < best_va_medr:
        txt_va_med_rank = col.colored(txt_va_med_rank, BColors.OKGREEN)

    # todo: print learning rate as well
    print("\nEpoch {} took {:.3f}s (patience: {})".format(metrics['epoch'][-1], time.monotonic() - epoch_now, patience))
    print('  ' + txt_tr_loss + txt_va_loss)
    print('  ' + txt_tr_dist + txt_va_dist)
    print('  ' + txt_tr_map + txt_va_map + ' | ' + txt_tr_med_rank + txt_va_med_rank)
