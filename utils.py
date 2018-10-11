import os
import numpy as np
import pandas as pd
import glob
import logging
import skimage
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image, ImageFont, ImageDraw
import scipy.cluster
logging.basicConfig(level=logging.INFO, format="[%(lineno)4s : %(funcName)-30s ] %(message)s")

# GLOBALS
img_width = 224             # For VGG16
img_height = 224            # For VGG16
img_channel = 3

### FUNCTIONS ###
def init_globals(fashion_dataset_path):
    input_shape = (img_width, img_height, img_channel)
    class_names = []
    missing = ['Sundress', 'Cape', 'Nightdress', 'Shirtdress']
    with open(fashion_dataset_path + 'Anno/list_category_cloth.txt') as f:
        next(f)
        next(f)
        for line in f:
            line = line.split()[0]
            if line not in missing:
                class_names.append(line)
    attr_names = []
    with open(os.path.join(fashion_dataset_path, 'Anno/list_attr_cloth.txt')) as f:
        next(f)
        next(f)
        for line in f:
            attr_names.append('-'.join(line.split()[:-1]))
    # logging.info('classes {} {}'.format(len(class_names), class_names))
    # logging.info('attributes {} {}'.format(len(attr_names), attr_names))
    return (class_names, input_shape, attr_names)

def get_attr300():
    attr300 = [730, 365, 513, 884, 495, 836, 596, 822, 254, 142, 212, 226, 353, 162, 310, 546, 717, 837, 335, 380, 196, 892, 568, 441, 705, 81, 760, 601, 993, 620, 181, 830, 577, 112, 969, 955, 935, 1, 640, 358, 831, 720, 282, 820, 337, 681, 933, 983, 470, 616, 292, 236, 878, 121, 781, 818, 437, 93, 695, 61, 239, 770, 268, 713, 688, 913, 204, 698, 186, 881, 839, 722, 565, 786, 457, 823, 50, 571, 0, 817, 413, 429, 560, 751, 692, 593, 574, 453, 287, 825, 207, 191, 880, 563, 237, 300, 368, 897, 944, 11, 800, 811, 133, 920, 409, 984, 24, 697, 676, 245, 754, 83, 14, 141, 841, 415, 325, 608, 276, 843, 99, 851, 815, 747, 862, 44, 988, 249, 543, 775, 139, 18, 653, 264, 208, 87, 231, 899, 321, 115, 699, 15, 764, 531, 48, 749, 272, 852, 42, 937, 986, 129, 184, 336, 648, 911, 116, 872, 309, 201, 624, 30, 638, 797, 512, 45, 802, 450, 306, 423, 360, 410, 218, 154, 958, 982, 683, 708, 854, 599, 774, 900, 70, 562, 108, 785, 544, 793, 960, 666, 468, 36, 909, 848, 784, 538, 480, 20, 124, 612, 38, 90, 931, 674, 649, 953, 330, 58, 628, 153, 691, 159, 27, 389, 189, 110, 723, 618, 943, 474, 47, 947, 104, 307, 73, 941, 39, 396, 77, 279, 701, 658, 262, 131, 777, 902, 971, 567, 324, 929, 482, 889, 619, 827, 948, 891, 296, 901, 416, 72, 328, 302, 875, 662, 694, 671, 234, 277, 678, 667, 840, 642, 597, 669, 907, 489, 146, 293, 799, 446, 974, 476, 397, 842, 5, 449, 756, 763, 869, 725, 203, 119, 210, 132, 444, 654, 611, 977, 174, 893, 987, 921, 789, 150, 967, 687, 991, 17, 930, 821]
    F = {0: 0, 1: 1, 2: 2, 3: 2, 4: 4, 5: 5, 6: 5, 7: 7, 8: 7, 9: 9, 10: 10, 11: 11, 12: 11, 13: 13, 14: 14, 15: 15, 16: 16, 17: 17, 18: 18, 19: 18, 20: 20, 21: 21, 22: 22, 23: 23, 24: 24, 25: 24, 26: 24, 27: 27, 28: 28, 29: 30, 30: 30, 31: 31, 32: 32, 33: 33, 34: 34, 35: 35, 36: 36, 37: 37, 38: 38, 39: 39, 40: 39, 41: 41, 42: 42, 43: 42, 44: 44, 45: 45, 46: 46, 47: 47, 48: 48, 49: 50, 50: 50, 51: 50, 52: 50, 53: 50, 54: 50, 55: 55, 56: 56, 57: 57, 58: 58, 59: 58, 60: 60, 61: 61, 62: 61, 63: 61, 64: 61, 65: 61, 66: 61, 67: 61, 68: 68, 69: 69, 70: 70, 71: 72, 72: 72, 73: 73, 74: 73, 75: 75, 76: 77, 77: 77, 78: 77, 79: 79, 80: 80, 81: 81, 82: 81, 83: 83, 84: 84, 85: 85, 86: 85, 87: 87, 88: 87, 89: 87, 90: 90, 91: 90, 92: 90, 93: 93, 94: 93, 95: 93, 96: 93, 97: 93, 98: 93, 99: 99, 100: 100, 101: 101, 102: 101, 103: 103, 104: 104, 105: 104, 106: 106, 107: 106, 108: 108, 109: 109, 110: 110, 111: 110, 112: 112, 113: 112, 114: 112, 115: 115, 116: 116, 117: 116, 118: 118, 119: 119, 120: 120, 121: 121, 122: 121, 123: 121, 124: 124, 125: 124, 126: 126, 127: 127, 128: 128, 129: 129, 130: 130, 131: 131, 132: 132, 133: 133, 134: 133, 135: 136, 136: 136, 137: 137, 138: 138, 139: 139, 140: 139, 141: 141, 142: 142, 143: 142, 144: 142, 145: 142, 146: 146, 147: 142, 148: 142, 149: 142, 150: 150, 151: 142, 152: 142, 153: 153, 154: 154, 155: 154, 156: 156, 157: 157, 158: 159, 159: 159, 160: 160, 161: 161, 162: 162, 163: 162, 164: 162, 165: 162, 166: 162, 167: 162, 168: 162, 169: 162, 170: 162, 171: 162, 172: 162, 173: 162, 174: 174, 175: 174, 176: 176, 177: 176, 178: 178, 179: 179, 180: 180, 181: 181, 182: 181, 183: 181, 184: 184, 185: 184, 186: 186, 187: 186, 188: 186, 189: 189, 190: 189, 191: 191, 192: 191, 193: 191, 194: 194, 195: 195, 196: 196, 197: 196, 198: 196, 199: 196, 200: 200, 201: 201, 202: 201, 203: 203, 204: 204, 205: 204, 206: 204, 207: 207, 208: 208, 209: 209, 210: 210, 211: 210, 212: 212, 213: 212, 214: 212, 215: 212, 216: 212, 217: 212, 218: 218, 219: 212, 220: 222, 221: 212, 222: 212, 223: 212, 224: 212, 225: 218, 226: 226, 227: 226, 228: 226, 229: 229, 230: 229, 231: 231, 232: 231, 233: 231, 234: 234, 235: 234, 236: 236, 237: 237, 238: 236, 239: 239, 240: 239, 241: 239, 242: 239, 243: 239, 244: 244, 245: 245, 246: 245, 247: 247, 248: 248, 249: 249, 250: 250, 251: 251, 252: 251, 253: 253, 254: 254, 255: 254, 256: 254, 257: 254, 258: 254, 259: 254, 260: 254, 261: 261, 262: 262, 263: 263, 264: 264, 265: 264, 266: 266, 267: 266, 268: 268, 269: 268, 270: 268, 271: 268, 272: 272, 273: 272, 274: 272, 275: 275, 276: 276, 277: 277, 278: 277, 279: 279, 280: 279, 281: 281, 282: 282, 283: 282, 284: 282, 285: 285, 286: 287, 287: 287, 288: 287, 289: 287, 290: 287, 291: 287, 292: 292, 293: 293, 294: 293, 295: 293, 296: 296, 297: 296, 298: 296, 299: 296, 300: 300, 301: 301, 302: 302, 303: 303, 304: 304, 305: 305, 306: 306, 307: 307, 308: 307, 309: 309, 310: 310, 311: 310, 312: 310, 313: 310, 314: 310, 315: 310, 316: 310, 317: 310, 318: 310, 319: 310, 320: 310, 321: 321, 322: 322, 323: 323, 324: 324, 325: 325, 326: 325, 327: 325, 328: 328, 329: 328, 330: 330, 331: 331, 332: 331, 333: 333, 334: 334, 335: 335, 336: 336, 337: 337, 338: 337, 339: 337, 340: 337, 341: 337, 342: 337, 343: 337, 344: 337, 345: 337, 346: 335, 347: 335, 348: 335, 349: 349, 350: 350, 351: 351, 352: 352, 353: 353, 354: 353, 355: 353, 356: 353, 357: 353, 358: 358, 359: 358, 360: 360, 361: 360, 362: 360, 363: 363, 364: 364, 365: 365, 366: 365, 367: 365, 368: 368, 369: 368, 370: 368, 371: 368, 372: 365, 373: 365, 374: 365, 375: 365, 376: 365, 377: 365, 378: 365, 379: 365, 380: 380, 381: 380, 382: 380, 383: 380, 384: 365, 385: 365, 386: 365, 387: 365, 388: 365, 389: 389, 390: 389, 391: 389, 392: 365, 393: 365, 394: 394, 395: 396, 396: 396, 397: 397, 398: 397, 399: 399, 400: 400, 401: 401, 402: 402, 403: 402, 404: 404, 405: 405, 406: 406, 407: 407, 408: 408, 409: 409, 410: 410, 411: 411, 412: 412, 413: 413, 414: 413, 415: 415, 416: 416, 417: 416, 418: 418, 419: 419, 420: 419, 421: 421, 422: 422, 423: 423, 424: 423, 425: 423, 426: 423, 427: 427, 428: 428, 429: 429, 430: 429, 431: 429, 432: 429, 433: 433, 434: 434, 435: 435, 436: 435, 437: 437, 438: 437, 439: 439, 440: 440, 441: 441, 442: 441, 443: 441, 444: 444, 445: 444, 446: 446, 447: 447, 448: 448, 449: 449, 450: 450, 451: 450, 452: 452, 453: 453, 454: 453, 455: 453, 456: 453, 457: 457, 458: 458, 459: 459, 460: 460, 461: 461, 462: 462, 463: 463, 464: 463, 465: 465, 466: 465, 467: 468, 468: 468, 469: 470, 470: 470, 471: 470, 472: 470, 473: 473, 474: 474, 475: 474, 476: 476, 477: 476, 478: 478, 479: 479, 480: 480, 481: 481, 482: 482, 483: 483, 484: 484, 485: 485, 486: 485, 487: 487, 488: 487, 489: 489, 490: 490, 491: 491, 492: 492, 493: 493, 494: 494, 495: 495, 496: 495, 497: 495, 498: 495, 499: 495, 500: 495, 501: 495, 502: 495, 503: 495, 504: 495, 505: 495, 506: 495, 507: 495, 508: 495, 509: 495, 510: 510, 511: 511, 512: 512, 513: 513, 514: 513, 515: 513, 516: 513, 517: 513, 518: 513, 519: 513, 520: 513, 521: 513, 522: 513, 523: 513, 524: 513, 525: 513, 526: 513, 527: 513, 528: 513, 529: 513, 530: 513, 531: 531, 532: 513, 533: 531, 534: 531, 535: 513, 536: 513, 537: 537, 538: 538, 539: 539, 540: 540, 541: 541, 542: 542, 543: 543, 544: 544, 545: 544, 546: 546, 547: 546, 548: 546, 549: 546, 550: 546, 551: 546, 552: 546, 553: 546, 554: 546, 555: 546, 556: 546, 557: 546, 558: 558, 559: 559, 560: 560, 561: 560, 562: 562, 563: 563, 564: 563, 565: 565, 566: 565, 567: 567, 568: 568, 569: 568, 570: 578, 571: 571, 572: 571, 573: 573, 574: 574, 575: 574, 576: 576, 577: 577, 578: 577, 579: 579, 580: 579, 581: 581, 582: 582, 583: 583, 584: 583, 585: 585, 586: 585, 587: 587, 588: 588, 589: 589, 590: 589, 591: 591, 592: 591, 593: 593, 594: 593, 595: 595, 596: 596, 597: 597, 598: 597, 599: 599, 600: 600, 601: 601, 602: 601, 603: 601, 604: 601, 605: 601, 606: 601, 607: 601, 608: 608, 609: 609, 610: 611, 611: 611, 612: 612, 613: 612, 614: 612, 615: 612, 616: 616, 617: 617, 618: 618, 619: 619, 620: 620, 621: 620, 622: 620, 623: 623, 624: 624, 625: 624, 626: 624, 627: 627, 628: 628, 629: 628, 630: 628, 631: 631, 632: 632, 633: 633, 634: 634, 635: 635, 636: 636, 637: 636, 638: 638, 639: 639, 640: 640, 641: 641, 642: 642, 643: 642, 644: 642, 645: 648, 646: 648, 647: 648, 648: 648, 649: 649, 650: 650, 651: 651, 652: 652, 653: 653, 654: 654, 655: 655, 656: 656, 657: 658, 658: 658, 659: 659, 660: 660, 661: 661, 662: 662, 663: 663, 664: 664, 665: 665, 666: 666, 667: 667, 668: 666, 669: 669, 670: 670, 671: 671, 672: 671, 673: 671, 674: 674, 675: 674, 676: 676, 677: 677, 678: 678, 679: 678, 680: 678, 681: 681, 682: 681, 683: 683, 684: 683, 685: 683, 686: 686, 687: 687, 688: 688, 689: 688, 690: 690, 691: 691, 692: 692, 693: 693, 694: 694, 695: 695, 696: 695, 697: 697, 698: 698, 699: 699, 700: 700, 701: 701, 702: 702, 703: 703, 704: 704, 705: 705, 706: 705, 707: 705, 708: 708, 709: 708, 710: 708, 711: 711, 712: 712, 713: 713, 714: 713, 715: 715, 716: 717, 717: 717, 718: 717, 719: 717, 720: 720, 721: 721, 722: 722, 723: 723, 724: 724, 725: 725, 726: 726, 727: 727, 728: 728, 729: 729, 730: 730, 731: 730, 732: 730, 733: 730, 734: 730, 735: 730, 736: 730, 737: 730, 738: 730, 739: 730, 740: 730, 741: 730, 742: 730, 743: 730, 744: 730, 745: 730, 746: 746, 747: 747, 748: 748, 749: 749, 750: 750, 751: 751, 752: 752, 753: 753, 754: 754, 755: 755, 756: 756, 757: 757, 758: 758, 759: 759, 760: 760, 761: 761, 762: 762, 763: 763, 764: 764, 765: 765, 766: 766, 767: 767, 768: 770, 769: 770, 770: 770, 771: 770, 772: 770, 773: 773, 774: 774, 775: 775, 776: 775, 777: 777, 778: 777, 779: 779, 780: 780, 781: 781, 782: 781, 783: 781, 784: 784, 785: 785, 786: 786, 787: 786, 788: 786, 789: 789, 790: 789, 791: 789, 792: 793, 793: 793, 794: 794, 795: 795, 796: 796, 797: 797, 798: 798, 799: 799, 800: 800, 801: 800, 802: 802, 803: 802, 804: 804, 805: 805, 806: 806, 807: 807, 808: 808, 809: 809, 810: 810, 811: 811, 812: 811, 813: 813, 814: 814, 815: 815, 816: 816, 817: 817, 818: 818, 819: 818, 820: 820, 821: 821, 822: 822, 823: 823, 824: 824, 825: 825, 826: 826, 827: 827, 828: 827, 829: 829, 830: 830, 831: 831, 832: 831, 833: 833, 834: 834, 835: 835, 836: 836, 837: 837, 838: 838, 839: 839, 840: 840, 841: 841, 842: 842, 843: 843, 844: 843, 845: 845, 846: 846, 847: 848, 848: 848, 849: 849, 850: 850, 851: 851, 852: 852, 853: 853, 854: 854, 855: 854, 856: 854, 857: 854, 858: 858, 859: 859, 860: 860, 861: 861, 862: 862, 863: 862, 864: 862, 865: 865, 866: 866, 867: 867, 868: 868, 869: 869, 870: 870, 871: 872, 872: 872, 873: 875, 874: 875, 875: 875, 876: 876, 877: 880, 878: 878, 879: 878, 880: 880, 881: 881, 882: 881, 883: 884, 884: 884, 885: 884, 886: 884, 887: 884, 888: 888, 889: 889, 890: 890, 891: 891, 892: 892, 893: 893, 894: 893, 895: 895, 896: 896, 897: 897, 898: 898, 899: 899, 900: 900, 901: 901, 902: 902, 903: 903, 904: 904, 905: 905, 906: 906, 907: 907, 908: 908, 909: 909, 910: 909, 911: 911, 912: 912, 913: 913, 914: 913, 915: 915, 916: 920, 917: 920, 918: 920, 919: 920, 920: 920, 921: 921, 922: 922, 923: 923, 924: 924, 925: 925, 926: 926, 927: 927, 928: 928, 929: 929, 930: 930, 931: 931, 932: 932, 933: 933, 934: 933, 935: 935, 936: 935, 937: 937, 938: 938, 939: 939, 940: 940, 941: 941, 942: 941, 943: 943, 944: 944, 945: 945, 946: 946, 947: 947, 948: 948, 949: 950, 950: 950, 951: 951, 952: 952, 953: 953, 954: 955, 955: 955, 956: 955, 957: 957, 958: 958, 959: 958, 960: 960, 961: 960, 962: 962, 963: 963, 964: 964, 965: 965, 966: 966, 967: 967, 968: 968, 969: 969, 970: 969, 971: 971, 972: 972, 973: 973, 974: 974, 975: 975, 976: 976, 977: 977, 978: 978, 979: 979, 980: 980, 981: 981, 982: 982, 983: 983, 984: 984, 985: 985, 986: 986, 987: 987, 988: 988, 989: 989, 990: 990, 991: 991, 992: 991, 993: 993, 994: 993, 995: 993, 996: 993, 997: 993, 998: 993, 999: 993}
    return (attr300, F)

def bb_intersection_over_union(boxes1, boxes2):
    x11, y11, x12, y12 = boxes1[0], boxes1[1], boxes1[0] + boxes1[2], boxes1[1] + boxes1[3]
    x21, y21, x22, y22 = boxes2[0], boxes2[1], boxes2[0] + boxes2[2], boxes2[1] + boxes2[3]
    xA = np.maximum(x11, x21)
    yA = np.maximum(y11, y21)
    xB = np.minimum(x12, x22)
    yB = np.minimum(y12, y22)
    interArea = np.maximum((xB - xA + 1), 0) * np.maximum((yB - yA + 1), 0)
    boxAArea = (x12 - x11 + 1) * (y12 - y11 + 1)
    boxBArea = (x22 - x21 + 1) * (y22 - y21 + 1)
    iou = interArea / (boxAArea + boxBArea - interArea)
    return iou

def plot_history(output_path, sep=';', show=False):
    def plot_history(metric, names, log):
        plt.style.use("ggplot")
        (fig, ax) = plt.subplots(len(names), 1, figsize=(8, 8))
        if len(names) == 1:
            ax = [ax]
        # loop over the accuracy names
        for (i, l) in enumerate(names):
            # plot the loss for both the training and validation data
            ax[i].set_title("{} for {}".format(metric, l))
            ax[i].set_xlabel("Epoch #")
            ax[i].set_ylabel(metric)
            ax[i].plot(log.index, log[l], label=l)
            ax[i].plot(log.index, log["val_" + l],
                       label="val_" + l)
            ymin = np.min([np.min(log["val_" + l][20:]), np.min(log[l][20:])])
            ymax = np.max([np.max(log["val_" + l][20:]), np.max(log[l][20:])])
            if len(log[l]) > 20:
                ax[i].set_ylim([ymin, ymax])
            ax[i].legend()
        # save the accuracies figure
        plt.tight_layout()
        # fig.show()
        fig.savefig(os.path.join(output_path, metric+'.png'))
    log = pd.DataFrame.from_csv(os.path.join(output_path, 'model_train.csv'), sep=sep)
    losses = [i for i in log.keys() if not i.startswith('val_') and i.endswith('loss')]
    errs = [i for i in log.keys() if not i.startswith('val_') and i.endswith('error')]
    accs = [i for i in log.keys() if not i.startswith('val_') and i.endswith('acc')]
    if errs:
        plot_history('Error', errs, log)
    if accs:
        plot_history('Accuracy', accs, log)
    plot_history('Loss', losses, log)
    if show:
        plt.show()
    return

def get_image_paths(prediction_path):
    images_path_name = sorted(glob.glob(prediction_path + '/*.*g'))
    if os.name == 'nt':
        images_path_name = [x.replace('\\', '/') for x in images_path_name]
    return images_path_name

def draw_rect(ax, img, gt_bbox, text=None, textcolor=(0,0,0), edgecolor='red',linewidth=4):
    def display_bbox_text(img, bbox, text, color=(0, 0, 0), fontsize=32):
        draw = ImageDraw.Draw(img)
        # font = ImageFont.truetype(<font-file>, <font-size>)
        # font = ImageFont.truetype("sans-serif.ttf", 16)
        # font = ImageFont.truetype("DroidSans.ttf", 16)
        # font = ImageFont.truetype('fonts/alterebro-pixel-font.ttf', 30)
        # font = ImageFont.truetype('/usr/share/fonts/truetype/ubuntu-font-family/Ubuntu-C.ttf', 16)
        # font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
        font = ImageFont.truetype('extra/fonts/Ubuntu-C.ttf', fontsize)
        draw.text((bbox[0], bbox[1]), text, color, font=font)
        # draw.text((bbox[0], bbox[1]), text,(255,0,0),font=font)
    x, y, w, h = gt_bbox[0], gt_bbox[1], gt_bbox[2], gt_bbox[3]
    rect = mpatches.Rectangle((x, y), w, h, fill=False, edgecolor=edgecolor, linewidth=linewidth)
    ax.add_patch(rect)
    if text is not None:
        display_bbox_text(img, gt_bbox, text, textcolor)
    ax.imshow(img, aspect='equal')

def get_validation_data(path):
    base = os.path.split(path)[0]
    if os.path.exists(os.path.join(base, 'validation.npz')):
        return np.load(os.path.join(base, 'validation.npz'))
    with open(path) as f:
        first = f.readline().rstrip()
        temp = np.load(os.path.join(base, first))
        keys = temp.keys()
        values = dict(((key, np.array(temp[key])) for key in keys))
        for line in f:
            temp = np.load(os.path.join(base, line.rstrip()))
            for key in keys:
                values[key] = np.concatenate([values[key], temp[key]])
        params = ','.join(['{}=values[{}]'.format(key, key) for key in keys])
        np.savez_compressed(open(os.path.join(base, 'validation.npz'), 'wb'), **values)
    return np.load(os.path.join(base, 'validation.npz'))

def double_bottleneck_batch(btl_path, num_per_file):
    btl_path_save = btl_path + '_new'
    btl_train_path = os.path.join(btl_path_save, 'train')
    btl_val_path = os.path.join(btl_path_save, 'validation')
    if not os.path.exists(btl_path_save):
        os.mkdir(btl_path_save)
    if not os.path.exists(btl_train_path):
        os.makedirs(btl_train_path)
    if not os.path.exists(btl_val_path):
        os.makedirs(btl_val_path)
    for train_val in ['validation', 'train']:
        save = False
        index = 0
        with open(os.path.join(btl_path_save, 'btl_' + train_val + '_npz.txt'), 'w') as fw:
            with open(os.path.join(btl_path, 'btl_' + train_val + '_npz.txt')) as fr:
                for line in fr:
                    temp = np.load(os.path.join(btl_path, line.rstrip()))
                    if save:
                        for key in keys:
                            values[key] = np.concatenate([values[key], temp[key]])
                        btl_save_file_name = train_val + '/btl_' + train_val + '_' + \
                                str(num_per_file) + '_' + str(index*num_per_file).zfill(7) + '.npz'
                        np.savez_compressed(open(os.path.join(btl_path_save, btl_save_file_name), 'wb'), **values)
                        fw.write(str(btl_save_file_name) + '\n')
                        index += 1
                        save = False
                    else:
                        keys = temp.keys()
                        values = dict(((key, temp[key]) for key in keys))
                        save = True

def change_bottleneck(btl_path, m, num_per_file):
    def save(values, keys, train_val,num_per_file, index, btl_path_save, fw):
        n = values[keys[0]].shape[0] // num_per_file
        for i in range(n):
            temp = dict(((key, values[key][i*num_per_file:(i+1)*num_per_file]) for key in keys))
            btl_save_file_name = train_val + '/btl_' + train_val + '_' + \
                                str(num_per_file) + '_' + str(index*num_per_file).zfill(7) + '.npz'
            np.savez_compressed(open(os.path.join(btl_path_save, btl_save_file_name), 'wb'), **temp)
            fw.write(str(btl_save_file_name) + '\n')
            index += 1
        return index

    btl_path_save = btl_path + '_new'
    btl_train_path = os.path.join(btl_path_save, 'train')
    btl_val_path = os.path.join(btl_path_save, 'validation')
    if not os.path.exists(btl_path_save):
        os.mkdir(btl_path_save)
    if not os.path.exists(btl_train_path):
        os.makedirs(btl_train_path)
    if not os.path.exists(btl_val_path):
        os.makedirs(btl_val_path)
    for train_val in ['validation', 'train']:

        index = 0
        with open(os.path.join(btl_path_save, 'btl_' + train_val + '_npz.txt'), 'w') as fw:
            with open(os.path.join(btl_path, 'btl_' + train_val + '_npz.txt')) as fr:
                files_read = 0
                for line in fr:
                    temp = np.load(os.path.join(btl_path, line.rstrip()))
                    if files_read == 0:
                        keys = temp.keys()
                        values = dict(((key, temp[key]) for key in keys))
                    else:
                        for key in keys:
                            values[key] = np.concatenate([values[key], temp[key]])
                    files_read += 1
                    if files_read < m:
                        continue
                    index = save(values, keys, train_val, num_per_file, index, btl_path_save, fw)
                    files_read = 0

                if files_read > 0:
                    save(values, keys, train_val, num_per_file, index, btl_path_save, fw)
                    
                    
if __name__ == '__main__':
    change_bottleneck('/media/star/3C4C65AA4C65601E/dev/deepfashion/fashion_data/bottleneck256_350', 7, 224)
    # btl_path = '/media/star/3C4C65AA4C65601E/dev/deepfashion/fashion_data/bottleneck128'
    # btl_path_save = btl_path + 'us'
    # btl_train_path = os.path.join(btl_path_save, 'train')
    # btl_val_path = os.path.join(btl_path_save, 'validation')
    # if not os.path.exists(btl_path_save):
    #     os.mkdir(btl_path_save)
    # if not os.path.exists(btl_train_path):
    #     os.makedirs(btl_train_path)
    # if not os.path.exists(btl_val_path):
    #     os.makedirs(btl_val_path)
    # for train_val in ['validation', 'train']:
    #     index = 0
    #     with open(os.path.join(btl_path_save, 'btl_x_' + train_val + '_npz.txt'), 'w') as fwx:
    #         with open(os.path.join(btl_path_save, 'btl_y_' + train_val + '_npz.txt'), 'w') as fwy:
    #             with open(os.path.join(btl_path, 'btl_' + train_val + '_npz.txt')) as fr:
    #                 for line in fr:
    #                     temp = np.load(os.path.join(btl_path, line.rstrip()))
    #                     btl = temp['btl']
    #                     pcbboxattr = temp['pcbboxattr']
    #                     btl_save_file_name = train_val + '/btl_x_' + train_val + '_' + \
    #                                 str(128) + '_' + str(index*128).zfill(7) + '.npy'
    #                     np.save(open(os.path.join(btl_path_save, btl_save_file_name), 'wb'), btl)
    #                     fwx.write(str(btl_save_file_name) + '\n')
    #                     btl_save_file_name = train_val + '/btl_y_' + train_val + '_' + \
    #                                 str(128) + '_' + str(index*128).zfill(7) + '.npy'
    #                     np.save(open(os.path.join(btl_path_save, btl_save_file_name), 'wb'), pcbboxattr)
    #                     fwy.write(str(btl_save_file_name) + '\n')
    #                     index += 1
    # temp = get_validation_data('fashion_data\\bottleneck\\btl_validation_npz.txt')
    # con = np.concatenate([temp['pc'].reshape((temp['pc'].shape[0], 1)), temp['bbox'], temp['attr']], axis=1)
    # keys = temp.keys()
    # values = dict(((key, np.array(temp[key])) for key in keys))
    # r=8
    # for i in ['1', '2']:
    #     fig0 = plt.figure(figsize=(5, 5), frameon=False)
    #     fig0.set_size_inches(5, 5)
    #     ax0 = plt.Axes(fig0, [0., 0., 1., 1.])
    #     ax0.set_axis_off()
    #     fig0.add_axes(ax0)
    #     img = Image.open('prediction/'+i+'.jpg')
    #     w,h = img.size[0], img.size[1]
    #     if w > h:
    #         d = (w-h)//2
    #         img=img.crop((d,0,w-d,h))
    #     else:
    #         d=(h-w)//2
    #         img=img.crop((0,d,w,h-d))
    #     ax0.imshow(img)
    #     plt.show()




