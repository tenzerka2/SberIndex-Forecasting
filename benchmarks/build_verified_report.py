"""Build the Russian review report from verified CI aggregates (optional reportlab dependency)."""
from pathlib import Path
import argparse
import json
from html import escape
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
NAMES = {
    'v3_hedge': 'V3-hedge', 'v3': 'V3', 'v4_diversified': 'V4-diversified',
    'prophet_auto': 'Prophet auto', 'prophet_log_fourier4': 'Prophet log + Fourier 4',
    'prophet_log_month_dummies': 'Prophet log + месяцы',
    'seasonal_naive': 'Сезонный наивный', 'national_yoy_fallback': 'Национальный резерв',
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', type=Path, default=ROOT/'outputs/prophet_verified')
    ap.add_argument('--font-dir', type=Path, required=True, help='Directory containing DejaVuSans.ttf and DejaVuSans-Bold.ttf')
    args = ap.parse_args()
    metrics = pd.read_csv(args.input/'metrics.csv')
    comparison = pd.read_csv(args.input/'comparisons.csv')
    provenance = json.loads((args.input/'provenance.json').read_text())
    assert set(NAMES).issubset(set(metrics.model)), 'All three Prophet variants must be complete'
    def mae(model, h):
        v = metrics.loc[(metrics.model == model) & (metrics.h == h), 'MAE']
        return f'{v.iloc[0]:.2f}' if len(v) else '-'
    forecast_table = [['Метод', '1 мес.', '3 мес.', '6 мес.', '12 мес.']]
    for model in ['v3_hedge', 'v3', 'v4_diversified', 'prophet_auto', 'prophet_log_fourier4', 'prophet_log_month_dummies', 'seasonal_naive', 'national_yoy_fallback']:
        forecast_table.append([NAMES[model]]+[mae(model,h) for h in [1,3,6,12]])
    comparison_table = [['Контрольная модель', '1 мес.', '3 мес.', '6 мес.']]
    for baseline in ['prophet_auto','prophet_log_fourier4','prophet_log_month_dummies']:
        rows=comparison[(comparison.model=='v3_hedge') & (comparison.baseline==baseline)].set_index('h')
        comparison_table.append([NAMES[baseline]]+[f'{rows.loc[h,"MAE_reduction_pct"]:.1f}%' for h in [1,3,6]])
    growth = metrics[metrics.model=='v3_hedge'].set_index('h')
    shock_table = [
        ['Задача / метод','AP','Precision','Recall'],
        ['Обнаружение: BOCPD','0.4123','0.445','0.373'],
        ['Обнаружение: jump','0.4065','0.450','0.377'],
        ['Обнаружение: logistic-base','0.4655','0.545','0.457'],
        ['Упреждение: jump','0.0552','0.105','0.128'],
        ['Упреждение: logistic-base','0.0264','0.030','0.037'],
    ]
    # A shared content structure keeps PDF and Markdown numbers and caveats aligned.
    pages = [
        [('title','Прогноз расходов муниципалитетов'),
         ('subtitle','Проверенные результаты и границы применимости | 7 октября 2026'),
         ('h','1. Сравнение с настоящим Prophet'),
         ('p','В GitHub Actions выполнены три варианта Prophet на одной фиксированной выборке 100 муниципалитетов. Таблица показывает MAE в рублях; меньше означает лучше. Исследуется категория «Все категории». Это наш воспроизводимый ретроспективный тест, а не оценка закрытой модели Сбера.'),
         ('table',forecast_table),
         ('p','На 1, 3 и 6 месяцах сравниваются 1000, 800 и 500 одинаковых пар соответственно. На 12 месяцах доступны только 100 пар с одной датой прогноза; V3/V4 там не определены. Национальный резерв - отдельный метод.'),
         ('h','Снижение MAE у V3-hedge относительно каждого Prophet'),
         ('table',comparison_table),
         ('p','Все три настройки Prophet заданы до серверного расчёта и опубликованы вместе. Неудачные прогнозы и варианты не удалялись. Выигрыш на уже изученной истории не гарантирует сохранения результата на новых данных.')],
        [('title','Метод и протокол проверки'),
         ('h','2. Почему выбрана V3-hedge'),
         ('p','Модель объединяет локальную сезонную компоненту и общую динамику панели 2016 муниципалитетов с равными весами. Для общего темпа роста используются муниципальные окна 2 и 6 месяцев и национальное окно 2 месяца. Затем добавляется поправка по двум уже наблюдаемым ошибкам. Обучение поправки в каждой дате использует только завершившиеся прогнозы.'),
         ('p','V4-diversified и robust-filtered-feedback проверены как улучшения. Более низкая ошибка отдельного окна не перенеслась на все проверки. V3-hedge выбран для устойчивого рабочего варианта по расширенным ретроспективным окнам; это исследовательский выбор после просмотра истории.'),
         ('h','3. Одинаковые цели, разные способы использования информации'),
         ('p','История муниципальных данных: январь 2023 - декабрь 2024. Выборка оцениваемых рядов фиксирована (seed=0). На h=1 используются origins февраль-ноябрь 2024; на h=3 февраль-сентябрь; на h=6 февраль-июнь. Для h=12 origin - декабрь 2023.'),
         ('p','Панельная модель видит прошлое всей панели; Prophet обучается отдельно на каждом ряду. Это сравнение методов, а не одинакового набора признаков. Национальные ряды обрезаются по месяцу наблюдения, но исторические версии публикаций отсутствуют. Отбор полных рядов также ретроспективный.'),
         ('p','Prophet: auto; логарифм и Fourier порядка 4; логарифм и 11 индикаторов месяца. У последних двух changepoint_prior_scale=0.01, у индикаторов prior_scale=0.1. Недельная и дневная сезонность отключены. Отрицательные прогнозы ограничиваются нулём. Горизонт означает значение месяца t+h, а не сумму за период.'),
         ('h','4. Высокий R² уровня не означает точный прогноз роста'),
         ('p',f'R² логарифмического годового роста у V3-hedge: h=1 {growth.loc[1,"R2_log_yoy"]:.3f}; h=3 {growth.loc[3,"R2_log_yoy"]:.3f}; h=6 {growth.loc[6,"R2_log_yoy"]:.3f}. На шести месяцах показатель отрицательный. Поэтому малую MAE относительно Prophet нельзя превращать в заявление о надёжном долгосрочном прогнозе темпов роста.'),
         ('p','12-месячный резерв умножает прошлогоднее муниципальное значение на доступный национальный годовой коэффициент. Только одна дата проверки не позволяет убедительно оценить его устойчивость.')],
        [('title','Структурные изменения и новости'),
         ('h','5. Разделяем обнаружение и упреждение'),
         ('p','Исправлена разметка событий: масштаб определяется по прошлому, сезонное эхо ищется только назад, соседние события подавляются причинно. Событие подтверждается через два месяца; метки будущих событий для горизонта 3 месяца созревают через пять. Неизвестные метки исключаются из обучения и оценки.'),
         ('p','Старая разметка меняла 214-523 созревшие метки при добавлении будущих данных. После исправления на всех проверенных origins изменений нет. Это устраняет найденную проблему созревания, но не доказывает отсутствие любых возможных утечек.'),
         ('table',shock_table),
         ('p','Здесь используются все 2016 муниципалитетов, отдельно от выборки Prophet. Бюджет сигналов фиксирован: 40 муниципалитетов в месяц (около 2%). AP - average precision. Logistic-base уменьшает ложные тревоги обнаружения до 0.903 на 100 ряд-месяцев; у jump 1.091. Это обнаружение уже начавшегося события, а не его предотвращение.'),
         ('p','Для будущих шоков лучший из показанных простых методов имеет precision лишь 10.5%: большинство предупреждений ложные. Надёжное упреждение и предотвращение экономических шоков не продемонстрированы.'),
         ('h','6. Новости проверены, но не улучшают итог'),
         ('p','После дедупликации есть 516 публикаций; консервативно сопоставлены 104 публикации с 77 муниципалитетами. Для 107 одноимённых рядов география не выдумывается. Используются публикации, доступные по дате, и признаки планируемых событий. Неполный архив и отсутствие официального crosswalk ограничивают анализ.'),
         ('p','Локальные новости почти не меняют AP обнаружения (0.4655 -> 0.4657) и не дают полезного упреждения. Национальная новостная поправка после исправления псевдорепликации ухудшает exact MAE до 1069.85 руб. Новости оставлены как исследовательский модуль; в основной прогноз не включены.')],
        [('title','Воспроизводимость и готовность'),
         ('h','7. Что сохранено для проверки'),
         ('p','В репозитории сохранены фактические прогнозы на всех 2400 парах, метрики, сравнения, SHA-256 входов и исходников, версии пакетов и ссылка на серверный запуск. Агрегатор отклоняет неполные результаты, разные пары, несовпадающие источники и версии. Четыре проверки протокола прошли локально и в Actions.'),
         ('p',f'Исходный commit расчёта: {provenance["source_commit"]}. GitHub Actions run: 37579679215. Python 3.11.16; NumPy 2.4.6; pandas 2.3.3; Prophet 1.5.0; cmdstanpy 1.3.0.'),
         ('p','Основные файлы: PROPHET_COMPARISON.md - протокол; outputs/prophet_verified/ - подтверждения; IMPROVEMENT_REPORT.md - аудит моделей и шоков; benchmarks/contest_prophet.py - расчёт; benchmarks/summarize_prophet_ci.py - независимый пересчёт метрик по сохранённым прогнозам.'),
         ('h','8. Что остаётся неподтверждённым'),
         ('p','Новая независимая временная выборка отсутствует: доступная история уже использовалась в исследовании. На длинных горизонтах мало дат. Современная выгрузка может содержать ревизии; календарь фактических публикаций не восстановлен. Преимущество над моделью организаторов не установлено: её прогнозы на этих же парах не предоставлены.'),
         ('p','TimesFM 2.5 присутствует в историческом эксперименте репозитория и тогда не улучшала точечный прогноз. В текущем серверном запуске foundation-модели не переобучались и не проверялись заново. Chronos-2 не запускалась. Старые оценки покрытия интервалов нельзя выдавать за подтверждённые на новом протоколе.'),
         ('p','Экспорт из декабря 2024 создаёт исторический пример прогнозов на 2025 год, а не актуальный прогноз из октября 2026. Для рабочего применения нужны свежие муниципальные данные и проверка фактической доступности национальных показателей.'),
         ('h','9. Формулировка для защиты'),
         ('p','Предлагается прозрачная панельная модель с воспроизводимым сравнением против трёх вариантов Prophet, причинной разметкой сдвигов и отдельным анализом новостей. Сильная сторона - качество краткосрочного прогноза и обнаружение уже начавшихся изменений. Основные открытые задачи - независимая проверка, длинные горизонты и полезное раннее предупреждение.')],
    ]
    md=[]
    for page in pages:
        for kind,value in page:
            if kind=='title': md.append('# '+value)
            elif kind=='h': md.append('## '+value)
            elif kind=='table':
                md.append('\n'.join(['| '+' | '.join(value[0])+' |', '| '+' | '.join(['---']*len(value[0]))+' |']+['| '+' | '.join(row)+' |' for row in value[1:]]))
            else: md.append(value)
    md.append('[Серверный запуск]('+provenance['workflow_run_url']+')')
    (ROOT/'VERIFIED_PROPHET_RESULTS.md').write_text('\n\n'.join(md)+'\n')
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.lib import colors
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
    pdfmetrics.registerFont(TTFont('DejaVu',str(args.font_dir/'DejaVuSans.ttf')))
    pdfmetrics.registerFont(TTFont('DejaVuBold',str(args.font_dir/'DejaVuSans-Bold.ttf')))
    styles={
        'title':ParagraphStyle('title',fontName='DejaVuBold',fontSize=21,leading=26,textColor=colors.HexColor('#123b36'),spaceAfter=15),
        'subtitle':ParagraphStyle('subtitle',fontName='DejaVu',fontSize=10,leading=15,textColor=colors.HexColor('#526963'),spaceAfter=15),
        'h':ParagraphStyle('h',fontName='DejaVuBold',fontSize=12,leading=17,spaceBefore=10,spaceAfter=7),
        'p':ParagraphStyle('p',fontName='DejaVu',fontSize=9,leading=13,spaceAfter=9),
        'cell':ParagraphStyle('cell',fontName='DejaVu',fontSize=8,leading=11),
    }
    out=ROOT/'output/pdf/SberIndex_verified_report.pdf';out.parent.mkdir(parents=True,exist_ok=True)
    story=[]
    for i,page in enumerate(pages):
        if i:story.append(PageBreak())
        for kind,value in page:
            if kind=='table':
                cells=[[Paragraph(escape(c),styles['cell']) for c in row] for row in value]
                widths=[175]+[(A4[0]-96-175)/(len(value[0])-1)]*(len(value[0])-1)
                table=Table(cells,colWidths=widths,repeatRows=1,hAlign='LEFT')
                table.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#dbeee6')),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.HexColor('#f4f8f6'),colors.white]),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7)]))
                story.extend([table,Spacer(1,10)])
            else:story.append(Paragraph(escape(value),styles[kind]))
    def footer(c,doc):
        c.setFont('DejaVu',8);c.setFillColor(colors.HexColor('#526963'))
        c.drawString(48,28,'SberIndex Forecasting | исследовательский результат | 07.10.2026')
        c.drawRightString(A4[0]-48,28,str(doc.page))
    SimpleDocTemplate(str(out),pagesize=A4,rightMargin=48,leftMargin=48,topMargin=42,bottomMargin=48,title='SberIndex: проверенные результаты',author='SberIndex Forecasting').build(story,onFirstPage=footer,onLaterPages=footer)
    print(out)


if __name__=='__main__':
    main()
