import torch
import torch.nn as nn
from torch import optim
import torch.utils.data as data
from utils_HSI import sample_gt, metrics, seed_worker,test_hsi,hsi_metrics
from datasets import get_dataset, HyperX
import random
import os
import time
import numpy as np
import pandas as pd
import argparse
from con_losses import SupConLoss, manifold_dis
from network import discrim_hyperG
from network import generator
from datetime import datetime

parser = argparse.ArgumentParser(description='PyTorch SDEnet')
parser.add_argument('--save_path', type=str, default='./results/')
parser.add_argument('--data_path', type=str, default='./Datasets/Pavia/')

parser.add_argument('--source_name', type=str, default='paviaU',
                    help='the name of the source dir')
parser.add_argument('--target_name', type=str, default='paviaC',
                    help='the name of the test dir')
parser.add_argument('--gpu', type=int, default=0,
                    help="Specify CUDA device (defaults to -1, which learns on CPU)")

group_train = parser.add_argument_group('Training')
group_train.add_argument('--patch_size', type=int, default=13,
                         help="Size of the spatial neighbourhood (optional, if ""absent will be set by the model)")
group_train.add_argument('--lr', type=float, default=1e-3,
                         help="Learning rate, set by the model if not specified.")
parser.add_argument('--momentum', type=float, default=0.9,
                    help='SGD momentum (default: 0.5)')
group_train.add_argument('--batch_size', type=int, default=256,
                         help="Batch size (optional, if absent will be set by the model")
group_train.add_argument('--pro_dim', type=int, default=128)
group_train.add_argument('--test_stride', type=int, default=1,
                         help="Sliding window step stride during inference (default = 1)")
parser.add_argument('--seed', type=int, default=1233,
                    help='random seed ')
parser.add_argument('--l2_decay', type=float, default=1e-4,
                    help='the L2  weight decay')
parser.add_argument('--num_epoch', type=int, default=400,
                    help='the number of epoch')
parser.add_argument('--training_sample_ratio', type=float, default=0.8,
                    help='training sample ratio')
parser.add_argument('--re_ratio', type=int, default=0,
                    help='multiple of of data augmentation')
parser.add_argument('--max_epoch', type=int, default=400)
parser.add_argument('--log_interval', type=int, default=40)
parser.add_argument('--d_se', type=int, default=64)
parser.add_argument('--lambda_1', type=float, default=1.0)
parser.add_argument('--lambda_2', type=float, default=1.0)
parser.add_argument('--lambda_3', type=float, default=0.1)
parser.add_argument('--lr_scheduler', type=str, default='none')
parser.add_argument('--low_freq', type=bool, default=False,
                    help="disturbed by low frequacy")

group_da = parser.add_argument_group('Data augmentation')
group_da.add_argument('--flip_augmentation', action='store_true', default=True,
                      help="Random flips (if patch_size > 1)")
group_da.add_argument('--radiation_augmentation', action='store_true', default=True,
                      help="Random radiation noise (illumination)")
group_da.add_argument('--mixture_augmentation', action='store_true', default=False,
                      help="Random mixes between spectra")
args = parser.parse_args()


def evaluate(net, val_loader, hyperparameter,gpu, tgt=False):
    ps = []
    ys = []
    for i, (x1, y1) in enumerate(val_loader):
        y1 = y1 - 1
        with torch.no_grad():
            x1 = x1.to(gpu)
            p1 = net(x1)
            p1 = p1.argmax(dim=1)
            ps.append(p1.detach().cpu().numpy())
            ys.append(y1.numpy())
    ps = np.concatenate(ps)
    ys = np.concatenate(ys)
    acc = np.mean(ys == ps) * 100
    if tgt:
        results = metrics(ps, ys, n_classes=ys.max() + 1)
        print(results['Confusion_matrix'], '\n', 'AA:', results['class_acc'], '\n', 'OA:', results['Accuracy'], '\n',
              'Kappa:', results['Kappa'])
        probility = test_hsi(net, val_loader.dataset.data, hyperparameter)
        np.save(args.source_name+'tsne_pred.npy',probility)
        prediction = np.argmax(probility, axis=-1)

        run_results = hsi_metrics(prediction, val_loader.dataset.label - 1, [-1],
                                  n_classes=hyperparameter['n_classes'])
        print(run_results)

        # with open('out_SDGnet_DG.log', 'a') as f:
        #     f.write("\n")
        #     f.write('OA:'+str(results['Accuracy'])+ "\n")
        #     f.write('AA:' + str(results['class_acc']) + "\n")
        #     f.write('Kappa:' + str(results['Kappa']) + "\n")
        #     f.write("\n")
        #
        # f.close()
        return acc, results, prediction
    else:
        return acc



def evaluate_tgt(cls_net, gpu, loader, hyperparameter,modelpath):
    saved_weight = torch.load(modelpath)
    cls_net.load_state_dict(saved_weight['Discriminator'])
    cls_net.eval()
    teacc, best_results,pred = evaluate(cls_net, loader, hyperparameter,gpu, tgt=True)
    return teacc, best_results,pred


def experiment():
    settings = locals().copy()
    print(settings)
    hyperparams = vars(args)
    print(hyperparams)
    now_time = datetime.now()
    time_str = datetime.strftime(now_time, '%m-%d_%H-%M-%S')
    root = os.path.join(args.save_path, args.source_name + 'to' + args.target_name)
    log_dir = os.path.join(root, str(args.lr) + '_dim' + str(args.pro_dim) +
                           '_pt' + str(args.patch_size) + '_bs' + str(args.batch_size) + '_' + time_str)
    if not os.path.exists(root):
        os.makedirs(root)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    # writer = SummaryWriter(log_dir)
    df = pd.DataFrame([args])
    df.to_csv(os.path.join(log_dir, 'params.txt'))

    seed_worker(args.seed)
    img_src, gt_src, LABEL_VALUES_src, IGNORED_LABELS, RGB_BANDS, palette = get_dataset(args.source_name,
                                                                                        args.data_path)
    img_tar, gt_tar, LABEL_VALUES_tar, IGNORED_LABELS, RGB_BANDS, palette = get_dataset(args.target_name,
                                                                                        args.data_path)
    sample_num_src = len(np.nonzero(gt_src)[0])
    sample_num_tar = len(np.nonzero(gt_tar)[0])
    n_classes = np.max(gt_src)
    print(n_classes)
    for i in range(n_classes):
        count_class = np.copy(gt_src)
        test_count = np.copy(gt_tar)
        # sparse_class=np.copy(sparse_ground_truth)

        count_class[(gt_src != i + 1)] = 0
        # sparse_class[(sparse_ground_truth != i + 1)[:H_SD, :W_SD]] = 0
        class_num = np.count_nonzero(count_class)

        test_count[gt_tar != i + 1] = 0

        print([i + 1], ':', class_num, np.count_nonzero(test_count))

    print("Total",np.count_nonzero(gt_src),np.count_nonzero(gt_tar))
    tmp = args.training_sample_ratio * args.re_ratio * sample_num_src / sample_num_tar
    num_classes = gt_src.max()
    N_BANDS = img_src.shape[-1]
    hyperparams.update({'n_classes': num_classes, 'n_bands': N_BANDS, 'ignored_labels': IGNORED_LABELS,
                        'device': args.gpu, 'center_pixel': None, 'supervision': 'full'})

    r = int(hyperparams['patch_size'] / 2) + 1
    img_src = np.pad(img_src, ((r, r), (r, r), (0, 0)), 'symmetric')
    img_tar = np.pad(img_tar, ((r, r), (r, r), (0, 0)), 'symmetric')
    gt_src = np.pad(gt_src, ((r, r), (r, r)), 'constant', constant_values=(0, 0))
    gt_tar = np.pad(gt_tar, ((r, r), (r, r)), 'constant', constant_values=(0, 0))


    if 'whu' in args.source_name:
        train_gt_src,val_gt_src, _, _ = sample_gt(gt_src, args.training_sample_ratio, mode='random')
        val_gt_src, _, _, _ = sample_gt(train_gt_src, 0.1, mode='random')
    else:
        train_gt_src,val_gt_src, _, _ = sample_gt(gt_src, args.training_sample_ratio, mode='random')

    # val_gt_src, _,_, _ = sample_gt(train_gt_src, 0.1, mode='random')
    print("All training number is,", np.count_nonzero(train_gt_src),np.count_nonzero(val_gt_src))
    test_gt_tar, _, _, _ = sample_gt(gt_tar, 1, mode='random')
    img_src_con, train_gt_src_con = img_src, train_gt_src
    val_gt_src_con = val_gt_src
    if tmp < 1:
        for i in range(args.re_ratio - 1):
            img_src_con = np.concatenate((img_src_con, img_src))
            train_gt_src_con = np.concatenate((train_gt_src_con, train_gt_src))
            val_gt_src_con = np.concatenate((val_gt_src_con, val_gt_src))

    hyperparams_train = hyperparams.copy()
    g = torch.Generator()
    g.manual_seed(args.seed)
    train_dataset = HyperX(img_src_con, train_gt_src_con, **hyperparams_train)
    train_loader = data.DataLoader(train_dataset,
                                   batch_size=hyperparams['batch_size'],
                                   pin_memory=True,
                                   worker_init_fn=seed_worker,
                                   generator=g,
                                   shuffle=True, )
    val_dataset = HyperX(img_src_con, val_gt_src_con, **hyperparams)
    val_loader = data.DataLoader(val_dataset,
                                 pin_memory=True,
                                 batch_size=hyperparams['batch_size'])
    test_dataset = HyperX(img_tar, test_gt_tar, **hyperparams)
    test_loader = data.DataLoader(test_dataset,
                                  pin_memory=True,
                                  worker_init_fn=seed_worker,
                                  generator=g,
                                  batch_size=hyperparams['batch_size'])
    imsize = [hyperparams['patch_size'], hyperparams['patch_size']]

    # D_net = discriminator.Discriminator(inchannel=N_BANDS, outchannel=args.pro_dim, num_classes=num_classes,
    #                                     patch_size=hyperparams['patch_size']).to(args.gpu)
    if 'whu' in args.source_name:
        pad = True
    else:
        pad = False
    D_net = discrim_hyperG.Discriminator(inchannel=N_BANDS, outchannel=args.pro_dim, num_classes=num_classes,
                                         patch_size=hyperparams['patch_size'],pad=pad).to(args.gpu)
    # D_net = vit_base_patch16(num_classes=num_classes, drop_path_rate=0.1, global_pool=True, img_size = hyperparams['patch_size'],
    #                          patch_size = 2,in_chans = N_BANDS,hidden_emb=args.pro_dim).to(args.gpu)
    # D_net = discriminator.D_Res_3d_CNN(in_channel=1, out_channel1=8, out_channel2=16, CLASS_NUM=num_classes,
    #                                    patch_size=hyperparams['patch_size'], n_bands=N_BANDS,embed_dim=args.pro_dim).to(args.gpu)
    D_opt = optim.Adam(D_net.parameters(), lr=args.lr)
    G_net = generator.Generator(n=args.d_se, imdim=N_BANDS, imsize=imsize, zdim=10, device=args.gpu,
                                low_freq=args.low_freq).to(args.gpu)
    G_opt = optim.Adam(G_net.parameters(), lr=args.lr)
    cls_criterion = nn.CrossEntropyLoss()
    con_criterion = SupConLoss(device=args.gpu)

    best_acc = 0
    best_acc_tgt = 0
    taracc, taracc_list = 0, []
    for epoch in range(1, args.max_epoch + 1):

        t1 = time.time()
        loss_list = []
        D_net.train()
        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(args.gpu), y.to(args.gpu)
            y = y - 1
            with torch.no_grad():
                x_ED = G_net(x)
            rand = torch.nn.init.uniform_(torch.empty(len(x), 1, 1, 1)).to(args.gpu)  # Uniform distribution

            x_tgt = G_net(x)
            x2_tgt = G_net(x)
            p_SD, z_SD = D_net(x, mode='train')
            p_ED, z_ED = D_net(x_ED, mode='train')

            x_ID = torch.zeros_like(x)

            for i, id in enumerate(y.unique()):
                mask = y == y.unique()[i]
                z_SD_i, z_ED_i = z_SD[mask], z_ED[mask]
                z_all_i = torch.cat([z_SD_i, z_ED_i], dim=0)
                range_i = range(0, z_all_i.size(0))

                idx_rand = random.sample(range_i, z_SD_i.size(0))

                p_dist_sd, c_dist_sd = manifold_dis(z_SD_i, z_all_i[idx_rand, :])
                p_dist_ed, c_dist_ed = manifold_dis(z_ED_i, z_all_i[idx_rand, :])

                lambda_pr1 = args.lambda_3 * p_dist_sd + c_dist_sd
                lambda_pd1 = args.lambda_3 * p_dist_ed + c_dist_ed
                x_ID[mask, :, :, :] = ((lambda_pd1 / (lambda_pr1 + lambda_pd1))) * x[mask, :, :, :] + (
                            lambda_pr1 / (lambda_pr1 + lambda_pd1)) * x_ED[mask, :, :, :]

            p_ID, z_ID = D_net(x_ID, mode='train')
            zsrc = torch.cat([z_SD.unsqueeze(1), z_ED.unsqueeze(1), z_ID.unsqueeze(1)], dim=1)
            # zsrc = torch.cat([z_SD.unsqueeze(1), z_ED.unsqueeze(1)], dim=1)
            src_cls_loss = cls_criterion(p_SD, y.long()) + cls_criterion(p_ED, y.long()) + cls_criterion(p_ID, y.long())
            p_tgt, z_tgt = D_net(x_tgt, mode='train')
            tgt_cls_loss = cls_criterion(p_tgt, y.long())

            # r1 = torch.randperm(hyperparams['batch_size'])
            # r2 = torch.randperm(hyperparams['batch_size'])
            # ml_real_out_shuffle = z_SD[r1[:, None]].view(z_SD.shape[0], z_SD.shape[-1])
            # ml_fake_out_shuffle = z_ED[r2[:, None]].view(z_ED.shape[0], z_ED.shape[-1])

            ############################
            p_dist, c_dist = manifold_dis(z_SD, z_ED)

            #
            # pd_r = pairwise_distances(z_SD, z_SD)
            # pd_f = pairwise_distances(z_ED, z_ED)

            # pd_r = pairwise_distances(F.normalize(p_SD), F.normalize(p_SD))
            # pd_f = pairwise_distances(F.normalize(p_ED), F.normalize(p_ED))

            # p_dist = torch.dist(pd_r, pd_f, 2)
            # c_dist = torch.dist(z_SD.mean(0), z_ED.mean(0), 2)
            # #
            g_loss = args.lambda_3 * p_dist + c_dist
            # g_loss = p_dist + c_dist
            # g_loss = 0.01*p_dist+(p_dist/g_dist)*c_dist

            zall = torch.cat([z_tgt.unsqueeze(1), zsrc], dim=1)
            con_loss = con_criterion(zall, y, adv=False)
            loss = src_cls_loss + args.lambda_1 * con_loss + tgt_cls_loss
            D_opt.zero_grad()
            loss.backward(retain_graph=True)

            num_adv = y.unique().size()
            # zsrc_con = torch.cat([z_tgt.unsqueeze(1), z_ED.unsqueeze(1), z_ID.unsqueeze(1)], dim=1)
            # zsrc_con = torch.cat([z_tgt.unsqueeze(1), z_ED.unsqueeze(1)], dim=1)
            zsrc_con = torch.cat([z_tgt.unsqueeze(1), z_ED.unsqueeze(1).detach(), z_ID.unsqueeze(1).detach()], dim=1)
            con_loss_adv = 0
            idx_1 = np.random.randint(0, zsrc.size(1))
            # z_SD_i, zsrc_i = z_SD[mask], zsrc_con[mask]
            # y_i = torch.cat([torch.zeros(z_SD_i.shape[0]), torch.ones(z_SD_i.shape[0])])
            zall = torch.cat([z_tgt.unsqueeze(1), zsrc[:, idx_1:idx_1 + 1].detach()], dim=1)
            con_loss_adv += con_criterion(zall, adv=True)

            # for i,id in enumerate(y.unique()):
            #     mask = y==y.unique()[i]
            #     z_SD_i, zsrc_i = z_SD[mask], zsrc_con[mask]
            #     y_i = torch.cat([torch.zeros(z_SD_i.shape[0]),torch.ones(z_SD_i.shape[0])])
            #     zall = torch.cat([z_SD_i.unsqueeze(1).detach(), zsrc_i[:,idx_1:idx_1+1]], dim=0)
            #     if y_i.size()[0] > 2:
            #         con_loss_adv += con_criterion(zall, y_i)
            # con_loss_adv = con_loss_adv/y.unique().shape[0]

            loss = tgt_cls_loss + args.lambda_1 * con_loss_adv + args.lambda_2 *g_loss
            G_opt.zero_grad()
            loss.backward()
            D_opt.step()
            G_opt.step()
            # if args.lr_scheduler in ['cosine']:
            #     scheduler.step()

            loss_list.append([src_cls_loss.item(), tgt_cls_loss.item(), con_loss.item(), con_loss_adv.item()])
        src_cls_loss, tgt_cls_loss, con_loss, con_loss_adv = np.mean(loss_list, 0)

        D_net.eval()
        teacc= evaluate(D_net, val_loader, hyperparams,args.gpu)
        if best_acc < teacc:
            best_acc = teacc
            torch.save({'Discriminator': D_net.state_dict()}, os.path.join(log_dir, f'best.pkl'))
        t2 = time.time()

        print(
            f'epoch {epoch}, train {len(train_loader.dataset)}, time {t2 - t1:.2f}, src_cls {src_cls_loss:.4f} tgt_cls {tgt_cls_loss:.4f} G_Loss {g_loss:.4f} con_adv {con_loss_adv:.4f} '
            f'/// p_dis {p_dist:.4f},p_dis {c_dist:.4f}, teacc {teacc:2.2f}')
        # writer.add_scalar('src_cls_loss', src_cls_loss, epoch)
        # writer.add_scalar('tgt_cls_loss', tgt_cls_loss, epoch)
        # writer.add_scalar('con_loss', con_loss, epoch)
        # writer.add_scalar('con_loss_adv', con_loss_adv, epoch)
        # writer.add_scalar('teacc', teacc, epoch)

        if epoch % args.log_interval == 0:
            pklpath = f'{log_dir}/best.pkl'
            taracc, result_tgt,pred = evaluate_tgt(D_net, args.gpu, test_loader, hyperparams,pklpath)
            if best_acc_tgt < taracc:
                best_acc_tgt = taracc
                best_results = result_tgt
                np.save(args.source_name+'_pred_OURS.npy', pred)
            taracc_list.append(round(taracc, 2))
            print(f'load pth, target sample number {len(test_loader.dataset)}, max taracc {max(taracc_list):2.2f}')

    with open('out_'+args.source_name+'_ablation.log', 'a') as f:
        f.write("\n")
        f.write('-----------------low_freq:' + str(args.low_freq) + "\n")
        f.write('max:' + str(max(taracc_list)) + "\n")
        f.write('LAMBDA1:' + str(args.lambda_1) + "\n")
        f.write('LAMBDA2:' + str(args.lambda_2) + "\n")
        f.write('LAMBDA3:' + str(args.lambda_3) + "\n")
        f.write('-----------------low_freq:' + str(args.low_freq) + "\n")
        f.write("\n")
        f.write('OA:' + str(best_results['Accuracy']) + "\n")
        f.write('AA:' + str(best_results['class_acc']) + "\n")
        f.write('Kappa:' + str(best_results['Kappa']) + "\n")
        f.write("\n")

    f.close()


if __name__ == '__main__':
    experiment()

