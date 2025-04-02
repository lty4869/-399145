import os
import pandas as pd

def to_timestamp_ignore_errors(x):
    try:
        return pd.to_datetime(x).timestamp()
    except Exception:
        return x


root = r'D:\Files\Documents\WeChat Files\wxid_v2md4i3fei6p22\FileStorage\File\2025-03\代码'


def translate_eng2chs(e: str):
    return e
    c = ''
    if e.startswith('HM') or e.startswith('ED'):
        c += '血肿' if e.startswith('HM') else '水肿'
        if e.endswith('volume'):
            return c + '体积'
        elif e.endswith('Ratio'):
            name, lr = '_'.join(e.split('_')[1:-2]), e.split('_')[-2]
            lr = '左' if lr == 'L' else '右'
            trans = {
                "ACA": "侧大脑前动脉",
                "MCA": "侧大脑中动脉",
                "PCA": "侧大脑后动脉",
                "Pons_Medulla": "侧脑桥/延髓",
                "Cerebellum": "侧小脑",
            }
            name = trans[name]
            return lr + name + c + '比例'
    return e



def _load_single_sheet(sheet_num):
    assert sheet_num in [1, 2, 3, 4, 5]
    num2path = {
        1: "竞赛发布数据/表1-患者列表及临床信息.xlsx",
        2: "竞赛发布数据/表2-患者影像信息血肿及水肿的体积及位置.xlsx",
        3: "竞赛发布数据/表3-患者影像信息血肿及水肿的形状及灰度分布.xlsx",
        4: "竞赛发布数据/表4-答案文件.xlsx",
        5: "竞赛发布数据/附表1-检索表格-流水号vs时间.xlsx",
    }
    num2path = {i: os.path.join(root, num2path[i]) for i in num2path}

    if sheet_num == 1:
        # 表1 sub074 131 132 首次流水号有误
        a = pd.read_excel(num2path[sheet_num])
        a.loc[73, '入院首次影像检查流水号'] = 20180719000020
        a.loc[130, '入院首次影像检查流水号'] = 20171220002173
        a.loc[131, '入院首次影像检查流水号'] = 20170107000727

        # 表1 将血压数据拆成2维
        a['收缩压'] = a['血压'].apply(lambda x: x.split('/')[0])
        a['舒张压'] = a['血压'].apply(lambda x: x.split('/')[1])
        a.drop('血压', axis=1, inplace=True)
        a = a.astype({'收缩压': "int", '舒张压': "int"})

        return a
    elif sheet_num == 3:
        return (pd.read_excel(num2path[sheet_num], sheet_name="ED"),
                pd.read_excel(num2path[sheet_num], sheet_name="Hemo"))
    elif sheet_num == 5:
        a = pd.read_excel(num2path[sheet_num])
        a = a[[i for i in a.columns if not i.startswith('Unnamed')]]
        return a
    else:
        return pd.read_excel(num2path[sheet_num])


def q1_a_data(first_100=True):
    sheet1 = _load_single_sheet(1)
    sheet2 = _load_single_sheet(2)
    sheet2 = sheet2[[i for i in sheet2.columns if not i.startswith("ED")]]
    sheetf1 = _load_single_sheet(5)
    for col in sheetf1.columns:
        if col.endswith('时间点'):
            sheetf1[col] = sheetf1[col].apply(to_timestamp_ignore_errors)
    if not first_100:
        sheet1 = sheet1[sheet1['Unnamed: 0'] < 'sub101']

    sheet = pd.merge(
        sheet1,
        sheet2[sheet2.columns.difference(sheet1.columns)],
        left_on='入院首次影像检查流水号',
        right_on='首次检查流水号'
    )
    sheet = pd.merge(
        sheet,
        sheetf1[sheetf1.columns.difference(sheet.columns)],
        left_on='入院首次影像检查流水号',
        right_on='入院首次检查流水号'
    )

    return sheet


def q1_b_data():
    sheet1 = _load_single_sheet(1)
    sheet2 = _load_single_sheet(2)
    _, sheet3_hemo = _load_single_sheet(3)
    sheet = pd.merge(
        sheet1,
        sheet2[sheet2.columns.difference(sheet1.columns)],
        left_on='入院首次影像检查流水号',
        right_on='首次检查流水号'
    )

    sheet = pd.merge(
        sheet,
        sheet3_hemo[sheet3_hemo.columns.difference(sheet.columns)],
        left_on='入院首次影像检查流水号',
        right_on='流水号',
        how='left'   # 131 132数据缺失
    )

    cols = (['Unnamed: 0'] + sheet1.columns[4:].tolist()
            + sheet2.columns[2:24].tolist()
            + sheet3_hemo.columns[2:].tolist())
    return sheet[cols]


def q2_ab_data():
    sheet2 = _load_single_sheet(2)
    sheet5 = _load_single_sheet(5)
    for col in sheet5.columns:
        if col.endswith('时间点'):
            sheet5[col] = sheet5[col].apply(to_timestamp_ignore_errors)

    sheet2 = sheet2[sheet2['ID'] < 'sub101']
    sheet = pd.merge(
        sheet2,
        sheet5,
        left_on='首次检查流水号',
        right_on='入院首次检查流水号',

    )
    # print(sheet.shape)
    ed_v = [col for col in sheet.columns if col.startswith('ED_volume')]
    # hao = [col for col in sheet.columns if '流水号' in col]
    shi = [col for col in sheet.columns if '时间点' in col]

    # 表2只有到第8次的流水号 因此水肿体积最多只有8次
    sheet = sheet[
        ['ID_x'] + ed_v + shi[:9]
    ]

    sheet.rename({
        "ID_x": "ID",
        "ED_volume": "ED_volume.0",
        "入院首次检查时间点": 'timestamp.0',
    }, inplace=True, axis=1)
    sheet.rename({f"随访{i}时间点": f'timestamp.{i}' for i in range(1, 9)}, inplace=True, axis=1)
    # for i in sheet.columns:
    #     print(i)
    # print(sheet.shape)
    # print(sheet)

    sheet1 = _load_single_sheet(1)
    for idx, _ in sheet.iterrows():
        before = sheet1.loc[idx, '发病到首次影像检查时间间隔']
        before *= 3600.
        start_stamp = sheet.loc[idx, 'timestamp.0']
        sheet.loc[idx, 'timestamp.0'] = before
        for i in range(1, 9):
            cur_stamp = sheet.loc[idx, f'timestamp.{i}']
            if not pd.isna(cur_stamp):
                sheet.loc[idx, f'timestamp.{i}'] = before + cur_stamp - start_stamp

    return sheet


def q2_c_data(which="ED"):
    sheet = q2_ab_data()
    sheet1 = _load_single_sheet(1)[
        ["Unnamed: 0",
         "脑室引流",
         "止血治疗",
         "降颅压治疗",
         "降压治疗",
         "镇静、镇痛治疗",
         "止吐护胃",
         "营养神经"]]
    sheet = pd.merge(
        sheet, sheet1, left_on='ID', right_on="Unnamed: 0"
    ).drop('Unnamed: 0', axis=1)

    # sheet = q2_d_data()

    for i in range(8):
        sheet["{}_volume_delta.{}".format(which, i + 1)] = \
            sheet["{}_volume.{}".format(which, i + 1)] - sheet[f"{which}_volume.0"]

    # print(sheet.columns)
    return sheet


def q2_d_data():
    sheet2 = _load_single_sheet(2)
    sheet5 = _load_single_sheet(5)
    for col in sheet5.columns:
        if col.endswith('时间点'):
            sheet5[col] = sheet5[col].apply(to_timestamp_ignore_errors)

    # sheet2 = sheet2[sheet2['ID'] < 'sub101']
    sheet = pd.merge(
        sheet2,
        sheet5,
        left_on='首次检查流水号',
        right_on='入院首次检查流水号',

    )

    ratio_dict = {col: f"{col}.0" for col in sheet.columns if col.endswith('Ratio')}
    ratio_dict.update({
        "ID_x": "ID",
        "ED_volume": "ED_volume.0",
        "HM_volume": "HM_volume.0",
        "入院首次检查时间点": 'timestamp.0',
    })

    sheet.rename(ratio_dict, inplace=True, axis=1)
    sheet.rename({f"随访{i}时间点": f'timestamp.{i}' for i in range(1, 9)}, inplace=True, axis=1)

    sheet1 = _load_single_sheet(1)
    for idx, _ in sheet.iterrows():
        before = sheet1.loc[idx, '发病到首次影像检查时间间隔']
        before *= 3600.
        start_stamp = sheet.loc[idx, 'timestamp.0']
        sheet.loc[idx, 'timestamp.0'] = before
        for i in range(1, 9):
            cur_stamp = sheet.loc[idx, f'timestamp.{i}']
            if not pd.isna(cur_stamp):
                sheet.loc[idx, f'timestamp.{i}'] = before + cur_stamp - start_stamp

    sheet = sheet
    sheet1 = _load_single_sheet(1)[
        ["Unnamed: 0",
         "脑室引流",
         "止血治疗",
         "降颅压治疗",
         "降压治疗",
         "镇静、镇痛治疗",
         "止吐护胃",
         "营养神经",
         "90天mRS",
         ]]

    sheet = pd.merge(
        sheet, sheet1, left_on='ID', right_on="Unnamed: 0"
    ).drop('Unnamed: 0', axis=1)

    sheet3_ed, sheet3_hemo = _load_single_sheet(3)
    sheet3_ed.drop("备注", axis=1, inplace=True)
    sheet3_hemo.drop("备注", axis=1, inplace=True)
    for idx, col_name in enumerate(['首次检查流水号'] + [f'随访{i}流水号_y' for i in range(1, 9)]):
        rename_dict = {col: f"{col}.{idx}" for col in sheet3_ed.columns}
        sheet = pd.merge(
            sheet, sheet3_ed.rename(rename_dict, axis=1), left_on=col_name, right_on=f"流水号.{idx}", how='left'
        )
    for idx, col_name in enumerate(['首次检查流水号'] + [f'随访{i}流水号_y' for i in range(1, 9)]):
        rename_dict = {col: f"{col}.{idx}" for col in sheet3_hemo.columns}
        sheet = pd.merge(
            sheet, sheet3_hemo.rename(rename_dict, axis=1), left_on=col_name, right_on=f"流水号.{idx}", how='left'
        )

    liushuihao_cols = [col for col in sheet.columns if '流水号' in col or '时间点' in col]
    liushuihao_cols += ['重复次数']
    sheet.drop(liushuihao_cols, axis=1, inplace=True)
    # print(sheet.shape)
    return sheet


if __name__ == "__main__":
    # print(_load_single_sheet(1))
    # q1_a_data()
    # q1_b_data()
    # q2_ab_data()
    # q2_c_data()
    q2_d_data()


