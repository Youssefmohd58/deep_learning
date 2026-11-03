"""Member 1 CUB data preparation and verification. No code runs at import time."""
from pathlib import Path
from collections import Counter, defaultdict
import hashlib, json, re
import numpy as np
import pandas as pd
from PIL import Image, ImageStat, ImageDraw
from tqdm.auto import tqdm


def save_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding='utf-8')


def scan_images(ds, name='train'):
    rows = []
    for i in tqdm(range(len(ds)), desc=f'Scan {name}'):
        item = {'split': name, 'row_index': i, 'decode_ok': False}
        try:
            image = ds[i]['image']; image.load()
            w, h = image.size
            gray = image.convert('L').resize((64,64))
            stat = ImageStat.Stat(gray)
            item.update(decode_ok=True, width=w, height=h, aspect_ratio=w/h,
                        mode=image.mode, mean_brightness=stat.mean[0], brightness_std=stat.stddev[0])
        except Exception as e:
            item['error'] = f'{type(e).__name__}: {e}'
        rows.append(item)
    return pd.DataFrame(rows)


def image_flags(meta):
    good = meta.loc[meta.decode_ok].copy()
    return good.loc[(good.mean_brightness < 40) | (good.brightness_std < 15) |
                    (good.width < 224) | (good.height < 224)].copy()


def contact_sheets(ds, flags, out_dir, per_sheet=48):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    for start in range(0, len(flags), per_sheet):
        selected = flags.iloc[start:start+per_sheet]
        canvas = Image.new('RGB', (8*150, 6*145), 'white'); draw = ImageDraw.Draw(canvas)
        for j, row in enumerate(selected.itertuples()):
            idx = int(row.row_index); x=(j%8)*150; y=(j//8)*145
            thumbnail=ds[idx]['image'].convert('RGB'); thumbnail.thumbnail((142,116))
            canvas.paste(thumbnail,(x+(150-thumbnail.width)//2,y))
            draw.text((x+5,y+120),f'row {idx}', fill='black')
        canvas.save(out_dir/f'flagged_contact_sheet_{start//per_sheet+1:02d}.jpg',quality=90)


def captions_and_flags(train, out_dir):
    out_dir=Path(out_dir); exported=[]; issues=[]; raw_total=0; removed_duplicate=0; removed_incomplete=0
    token_counts=Counter()
    for i, rec in enumerate(tqdm(train, desc='Inspect captions')):
        lines=[re.sub(r'\s+',' ',s).strip() for s in str(rec['description']).splitlines()]
        originals=[s for s in lines if s]
        raw_total+=len(originals); seen=set(); result=[]; repeats=0; short=0
        for s in originals:
            key=s.casefold()
            if key == 'this':
                removed_incomplete+=1; short+=1; continue
            if key in seen:
                removed_duplicate+=1; repeats+=1; continue
            seen.add(key); result.append(s)
            token_counts.update(re.findall(r"[a-z]+(?:'[a-z]+)?", s.casefold()))
        if repeats: issues.append({'row_index':i,'issue':'repeated_caption','detail':repeats})
        if short: issues.append({'row_index':i,'issue':'very_short_caption','detail':short})
        if len(originals)!=10: issues.append({'row_index':i,'issue':'unexpected_caption_count','detail':len(originals)})
        exported.append({'row_index':i, 'id':rec.get('id'), 'file_name':rec.get('file_name'),
                         'label':int(rec['label']), 'captions':result})
    path=out_dir/'clean_train_captions.jsonl'
    with path.open('w', encoding='utf-8') as fp:
        for rec in exported: fp.write(json.dumps(rec,ensure_ascii=False)+'\n')
    pd.DataFrame(issues,columns=['row_index','issue','detail']).to_csv(out_dir/'05_caption_review_flags.csv',index=False)
    return dict(image_records=len(exported), original_captions=raw_total,
                cleaned_captions=sum(len(r['captions']) for r in exported),
                duplicate_captions_removed=removed_duplicate,
                incomplete_captions_removed=removed_incomplete,
                cleaned_vocabulary_size=len(token_counts), review_flags=len(issues))


def make_manifest(train, out_dir, seed=42):
    from sklearn.model_selection import train_test_split
    path=Path(out_dir)/'train_validation_split.csv'
    labels=np.array(train['label']); expected=set(range(len(train)))
    if path.exists():
        old=pd.read_csv(path)
        if {'row_index','split','label'}.issubset(old.columns) and len(old)==len(train) and \
           set(old.row_index.astype(int))==expected and set(old.split)=={'train','validation'} and \
           all(str(labels[int(r.row_index)])==str(r.label) for r in old.itertuples()) and \
           old.row_index.is_unique:
            return old
    a,b=train_test_split(np.arange(len(train)),test_size=0.2, stratify=labels,random_state=seed)
    split_map={int(i):'train' for i in a}; split_map.update({int(i):'validation' for i in b})
    df=pd.DataFrame({'row_index':np.arange(len(train)), 'split':[split_map[i] for i in range(len(train))],
                    'label':labels, 'file_name':train['file_name']})
    df.to_csv(path,index=False)
    return df


def check_exact_duplicates(train, test, manifest, out_dir):
    out_dir=Path(out_dir); groups=defaultdict(list); failures=[]
    lookup=dict(zip(manifest.row_index.astype(int),manifest.split))
    for source, ds in [('original_train',train),('test',test)]:
        for i in tqdm(range(len(ds)),desc=f'Hash {source}'):
            try:
                image=ds[i]['image'].convert('RGB')
                h=hashlib.sha256(); h.update(str(image.size).encode()); h.update(image.tobytes())
                groups[h.hexdigest()].append((source,i))
            except Exception as e: failures.append({'source':source,'row_index':i,'error':str(e)})
    duplicates=[v for v in groups.values() if len(v)>1]
    edits=[]
    # Repair only train-vs-validation duplicates of the SAME label by swapping an unused training
    # example from that class into validation. Raise on cross-test duplicates: test must remain fixed.
    for entries in duplicates:
        splits={('test' if source=='test' else lookup[idx]) for source,idx in entries}
        if 'test' in splits and len(splits)>1:
            raise RuntimeError(f'Exact duplicate across official train and test: {entries}')
        if splits=={'train','validation'}:
            idxs=[i for source,i in entries if source=='original_train']
            labels={train[i]['label'] for i in idxs}
            if len(labels)!=1: raise RuntimeError(f'Conflicting duplicate labels: {entries}')
            klass=next(iter(labels))
            for val_idx in [i for i in idxs if lookup[i]=='validation']:
                occupied={idx for group in duplicates for src,idx in group if src=='original_train'}
                options=manifest.loc[(manifest.split=='train') & (manifest.label==klass) &
                                     (~manifest.row_index.isin(occupied)), 'row_index']
                if options.empty: raise RuntimeError('No nonduplicate class-matched replacement image')
                replacement=int(options.iloc[0]); manifest.loc[manifest.row_index==val_idx,'split']='train'
                manifest.loc[manifest.row_index==replacement,'split']='validation'
                lookup[val_idx]='train'; lookup[replacement]='validation'
                edits.append({'duplicate_validation_row':val_idx,'replacement_validation_row':replacement})
    cross=[v for v in duplicates if len({('test' if source=='test' else lookup[i]) for source,i in v})>1]
    if cross: raise RuntimeError(f'Unresolved cross-split duplicates: {cross}')
    if edits:
        path=out_dir/'train_validation_split.csv'; backup=out_dir/'train_validation_split_before_duplicate_fix.csv'
        if path.exists() and not backup.exists(): backup.write_bytes(path.read_bytes())
        assert len(manifest)==len(train)
        assert manifest.loc[manifest.split=='train','label'].nunique()==200
        assert manifest.loc[manifest.split=='validation','label'].nunique()==200
        manifest.to_csv(path,index=False)
    report={'exact_duplicate_groups':len(duplicates),'cross_split_groups_after_fix':len(cross),
            'decoding_failures':failures,'duplicate_groups':duplicates,'split_repairs':edits}
    save_json(out_dir/'final_audit'/'exact_pixel_audit.json',report)
    return report,manifest


def verify_loaders(loaders, expected, audit_dir):
    import torch
    results={}
    for name,loader,n in zip(['training','validation','test'],loaders,expected):
        total=0; seen=[]
        for batch in tqdm(loader,desc=f'Verifying {name}'):
            x=batch['image']; y=batch['label']
            assert x.ndim==4 and x.shape[1:]==(3,224,224)
            assert torch.isfinite(x).all() and bool(((y>=0)&(y<200)).all())
            total+=len(y);seen.extend(batch['row_index'].tolist())
        assert total==n and len(seen)==len(set(seen)),f'{name}: incomplete or repeated rows'
        results[name]={'passed':True,'images':total,'batches':len(loader)}
    save_json(Path(audit_dir)/'full_loader_test_results.json',results)
    return results


def save_review(flags, audit_dir):
    audit_dir=Path(audit_dir);path=audit_dir/'image_review_decisions.csv'
    if path.exists():
        df=pd.read_csv(path)
        if 'row_index' in df and 'decision' in df and set(df.row_index)==set(flags.row_index):
            return df
    df=flags.copy(); df['decision']='pending'; df['reason']='Requires contact-sheet review'
    df.to_csv(path,index=False)
    return df
