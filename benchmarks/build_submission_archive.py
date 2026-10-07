"""Build presentation evidence figures and a portable archive from versioned project files."""
from pathlib import Path
import hashlib,json,os,shutil,subprocess,zipfile
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'output/delivery/mplcache'))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    out=ROOT/'output/delivery';out.mkdir(parents=True,exist_ok=True)
    stage=out/'SberIndex_Submission_2026-10-07';stage.mkdir(exist_ok=True)
    figures=stage/'03_CHARTS';figures.mkdir(exist_ok=True)
    tables=stage/'04_TABLES';tables.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
    colors=['#155E55','#858C92','#D39454']
    def save(fig,name):
        fig.savefig(figures/(name+'.png'),dpi=220,bbox_inches='tight',facecolor='white')
        fig.savefig(figures/(name+'.svg'),bbox_inches='tight',facecolor='white')
        plt.close(fig)
    metrics=pd.read_csv(ROOT/'outputs/timesfm_summary/metrics.csv')
    comparison=[]
    for h in [1,3,6]:
        g=metrics[(metrics.h==h)&metrics.regime.eq('main')]
        comparison.append({'h':h,'V3 Hedge':float(g[g.model.eq('v3_hedge')].MAE.iloc[0]),
            'Лучший проверенный Prophet':float(g[g.model.str.startswith('prophet_')].MAE.min()),
            'TimesFM YoY':float(g[g.model.eq('timesfm_yoy')].MAE.iloc[0])})
    comparison=pd.DataFrame(comparison);comparison.to_csv(tables/'chart_comparison_mae.csv',index=False)
    fig,ax=plt.subplots(figsize=(11,6));x=np.arange(3)
    for j,name in enumerate(comparison.columns[1:]):
        bars=ax.bar(x+(j-1)*.24,comparison[name],width=.22,label=name,color=colors[j])
        ax.bar_label(bars,labels=[f'{v:.0f}' for v in comparison[name]],padding=4,fontsize=11)
    ax.set(xticks=x,xticklabels=['1 месяц','3 месяца','6 месяцев'],ylabel='MAE, рубли',ylim=(0,2700),title='Ошибка прогноза на 2016 муниципалитетах')
    ax.legend(frameon=False,ncol=3,loc='upper left',fontsize=10);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    fig.text(.12,.015,'Одинаковые прогнозные пары. Лучшая конфигурация Prophet выбрана отдельно для каждого горизонта.\nРетроспективная оценка; меньшая ошибка лучше.',fontsize=9,color='#555555');fig.subplots_adjust(bottom=.18);save(fig,'comparison_mae')
    intervals=pd.read_csv(ROOT/'outputs/timesfm_summary/intervals.csv');g=intervals[intervals.model.eq('timesfm_yoy')].sort_values('h')
    fig,ax=plt.subplots(figsize=(10,5.5));bars=ax.bar(x,100*g.coverage,color=colors[0],width=.5)
    ax.bar_label(bars,labels=[f'{v*100:.2f}%' for v in g.coverage],padding=-24,color='white')
    ax.axhline(80,ls='--',color='#B4533D',label='Номинальное покрытие 80%')
    ax.set(xticks=x,xticklabels=['1 месяц','3 месяца','6 месяцев'],ylim=(0,100),ylabel='Фактическое покрытие, %',title='Интервалы TimesFM YoY: проверка покрытия')
    ax.legend(frameon=False,loc='lower left');fig.text(.12,.02,'Эти интервалы относятся к TimesFM, а не к V3 Hedge. Ретроспективная оценка.',fontsize=10);fig.subplots_adjust(bottom=.16);save(fig,'coverage')
    warning=pd.read_csv(ROOT/'outputs/submission/warning_metrics.csv').set_index('task')
    fig,axes=plt.subplots(1,2,figsize=(11,5.5))
    for ax,task,title in zip(axes,['detect','predict'],['Обнаружение: logistic_base','Упреждение: jump']):
        row=warning.loc[task];a=int(row.true_positives);b=int(row.false_positives)
        ax.barh([0],[a],color=colors[0],label='Верные сигналы');ax.barh([0],[b],left=[a],color='#C5C9CB',label='Ложные сигналы')
        ax.text(a/2,0,str(a),ha='center',va='center',color='white',weight='bold');ax.text(a+b/2,0,str(b),ha='center',va='center',weight='bold')
        ax.set(title=title,yticks=[],xlabel='Количество сигналов',ylim=(-1,1));ax.text(.02,.83,f'Precision: {100*row.precision:.1f}%\nRecall: {100*row.recall:.1f}%',transform=ax.transAxes)
        ax.legend(frameon=False,loc='lower left',fontsize=10)
    fig.text(.08,.025,'Лимит 40 сигналов в месяц. Обнаружение — 10 месяцев; упреждение — 5 месяцев.\nЕдиница подсчёта — территория × месяц, не уникальное событие.',fontsize=10);fig.subplots_adjust(bottom=.21,wspace=.3);save(fig,'warning_outcomes')
    cases=pd.read_csv(ROOT/'outputs/submission/warning_cases.csv');history=pd.read_csv(ROOT/'outputs/submission/case_history.csv',parse_dates=['date'])
    fig,axes=plt.subplots(3,1,figsize=(11,10),sharex=True)
    for ax,kind,title in zip(axes,['hit','miss','false_alarm'],['Обнаружено','Пропущено в указанном месяце','Ложная тревога по разметке']):
        row=cases[cases.task.eq('detect')&cases['case'].eq(kind)].iloc[0];g=history[history.series.eq(row.series)]
        ax.plot(g.date,g.relative_log_level,color=colors[0],lw=2)
        ax.axvline(pd.Timestamp(row.signal_month),color='#D39454',ls='--',label='Месяц оценки / сигнала')
        if pd.notna(row.event_onset):ax.axvline(pd.Timestamp(row.event_onset),color='#858C92',ls=':',label='Размеченное начало')
        ax.set_title(f'{title}: {row["name"].strip()} (ID {row.territory_id})',fontsize=11,loc='left')
        ax.set_ylabel('Относительный\nлогарифм расходов',fontsize=10);ax.grid(alpha=.15);ax.legend(frameon=False,fontsize=9,loc='best')
    fig.text(.12,.015,'Ряды относительно медианы панели. Примеры выбраны по фиксированному правилу; внешние причины не установлены.',fontsize=9)
    fig.tight_layout(rect=(0,.04,1,1));save(fig,'detection_cases')
    for source in ['outputs/prophet_full_summary/metrics.csv','outputs/prophet_full_summary/comparisons.csv','outputs/timesfm_summary/metrics.csv','outputs/timesfm_summary/comparisons.csv','outputs/timesfm_summary/intervals.csv','outputs/submission/warning_metrics.csv','outputs/submission/warning_cases.csv','outputs/submission/case_history.csv']:
        path=Path(source);shutil.copyfile(ROOT/path,tables/(path.parent.name+'_'+path.name))
    shutil.copyfile(ROOT/'PRESENTATION_MATERIALS_CURRENT.md',stage/'01_PRESENTATION_MATERIALS.md')
    shutil.copyfile(ROOT/'SUBMISSION_CURRENT.md',stage/'02_CURRENT_REPORT.md')
    start='''СБЕРИНДЕКС: КОМПЛЕКТ ОТ 7 ОКТЯБРЯ 2026

Начните с 01_PRESENTATION_MATERIALS.md: содержание 14 слайдов, пояснения для выступления, полные числа и ответы на вопросы.
02_CURRENT_REPORT.md — единое описание проекта.
03_CHARTS — готовые графики PNG и редактируемые SVG, построенные из проверенных CSV.
04_TABLES — таблицы с точными значениями для оформления.
05_CODE — версия проекта с кодом, подготовленными данными, прогнозными парами и инструкциями.

Внутри 05_CODE сохранены исторические отчёты для аудита. Для презентации используйте только CURRENT-документы и актуальные полные метрики.
Архив не содержит исходных больших CSV, весов TimesFM, окружения Python или секретов. Для быстрой проверки они не требуются.
Для полной исследовательской цепочки и новых тяжёлых расчётов смотрите инструкции соответствующих скриптов.

Проверка: в папке 05_CODE установить numpy==2.4.6 и pandas==2.3.3, затем выполнить bash verify_submission.sh.
Презентация PPTX/PDF в архив не входит: содержание и графики подготовлены для самостоятельного оформления.
Перед подачей сверить регламент, срок и доступ жюри. Репозиторий частный; доступ автоматически не менялся.
'''
    (stage/'00_START_HERE.txt').write_text(start)
    names=subprocess.check_output(['git','ls-files','-z'],cwd=ROOT).decode().split('\0')
    for name in names:
        if not name:continue
        src=ROOT/name
        if src.is_file():
            dst=stage/'05_CODE'/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst)
    manifest={str(p.relative_to(stage)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(stage.rglob('*')) if p.is_file() and p.name!='CHECKSUMS.json'}
    (stage/'CHECKSUMS.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    archive=out/'SberIndex_Submission_2026-10-07.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(stage.rglob('*')):
            if p.is_file():z.write(p,str(Path(stage.name)/p.relative_to(stage)))
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None:raise ValueError('Broken ZIP')
        for name,expected in manifest.items():
            if hashlib.sha256(z.read(stage.name+'/'+name)).hexdigest()!=expected:raise ValueError('Archive checksum mismatch')
    print(archive);print('Archive verified:',archive.stat().st_size,'bytes;',len(manifest),'files')


if __name__=='__main__':main()
