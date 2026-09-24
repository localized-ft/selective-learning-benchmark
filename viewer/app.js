'use strict';
const $ = id => document.getElementById(id);
const names = {sft:'Standard SFT', vanilla:'Vanilla', kld:'KL regularization', ip:'Inoculation prompting', freeze_first:'Freeze first', freeze_second:'Freeze middle', freeze_last:'Freeze last'};
const colors = {sft:'#536a8c', vanilla:'#9b7549', kld:'#2558bd', ip:'#188678', freeze_first:'#9264b0', freeze_second:'#c16544', freeze_last:'#ae5184'};
const tasks = {bad_medical_advice:'Bad medical advice', risky_financial_advice:'Risky financial advice', school_of_reward_hacks:'School of reward hacks', good_vs_bad_mixed_multifact:'Mixed good / bad facts', target_only_no_hallucination:'Target facts / hallucination', german_city_names:'Old German city names', old_bird_names:'Old bird names'};
const models = {qwen3_32b:'Qwen3-32B', qwen3_8b:'Qwen3-8B', llama31_8b:'Llama-3.1-8B', olmo3_7b:'Olmo-3-7B'};
let data, selectedMethods = new Set(), points=[], offset=0, total=0, activeId='', requestVersion=0, detailVersion=0, tab='overview';
const nf = x => Number(x).toLocaleString('en-US');
const pct = x => x == null || !Number.isFinite(x) ? '—' : (x*100).toFixed(1)+'%';
const mean = a => {a=a.filter(Number.isFinite);return a.length?a.reduce((x,y)=>x+y,0)/a.length:null;};
function el(tag,text,cls){const e=document.createElement(tag);if(text!==undefined)e.textContent=text;if(cls)e.className=cls;return e;}
function dot(method){const e=el('span',undefined,'dot');e.style.backgroundColor=colors[method]||'#66758a';return e;}
function option(value,label){const o=el('option',label);o.value=value;return o;}
function options(id, values, labels, preferred){const s=$(id),prev=preferred??s.value;s.replaceChildren(...values.map(v=>option(v,labels[v]||v)));s.value=values.includes(prev)?prev:values[0];}
function error(e){$('error').textContent=e.message||String(e);$('error').hidden=false;}
function clearError(){$('error').hidden=true;}
async function api(path,q={}){const r=await fetch(path+'?'+new URLSearchParams(q));const d=await r.json();if(!r.ok)throw Error(d.error||'Unable to load results');return d;}
function toast(s){$('toast').textContent=s;$('toast').hidden=false;setTimeout(()=>$('toast').hidden=true,2400);}
function download(name,value,mime='application/json'){const a=el('a');const url=URL.createObjectURL(new Blob([value],{type:mime}));a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
function scopeRuns(){return data.runs.filter(r=>r.cohort===$('cohort').value&&r.task===$('task').value&&r.model===$('model').value);}
function eligibleRuns(){return scopeRuns().filter(r=>selectedMethods.has(r.method)&&($('seed').value==='all'||r.seed===null||String(r.seed)===$('seed').value));}
function runLabel(r){return `${names[r.method]||r.method} · ${r.seed===null?'reference':'seed '+r.seed}`;}
function syncOptions(reset=false){
 const cohort=$('cohort').value, runs=data.runs.filter(r=>r.cohort===cohort);
 options('task',[...new Set(runs.map(r=>r.task))],tasks,reset?'risky_financial_advice':undefined);
 options('model',[...new Set(runs.filter(r=>r.task===$('task').value).map(r=>r.model))],models);
 const scope=scopeRuns();const seeds=[...new Set(scope.map(r=>r.seed).filter(x=>x!==null))].sort();
 options('seed',['all',...seeds.map(String)],Object.fromEntries([['all','All available seeds'],...seeds.map(s=>[s,'Seed '+s]) ]));
 const methods=[...new Set(scope.map(r=>r.method))];if(reset||!methods.some(m=>selectedMethods.has(m)))selectedMethods=new Set(methods);
 $('methods').replaceChildren(...methods.map(m=>{const l=el('label',undefined,'method-option'),i=el('input');i.type='checkbox';i.checked=selectedMethods.has(m);i.addEventListener('change',()=>{i.checked?selectedMethods.add(m):selectedMethods.delete(m);refresh();});l.append(i,dot(m),el('span',names[m]||m));return l;}));
}
function computePoints(){
 const group=new Map();for(const r of eligibleRuns()){if(!group.has(r.method))group.set(r.method,[]);group.get(r.method).push(r);}
 const variant=$('variant').value;
 return [...group].map(([method,runs])=>{
  const values=runs.map(r=>r.variants[variant]);
  const p={method,runs,cap:mean(values.map(v=>v.capability)),ug:mean(values.map(v=>v.unwanted_generalization)),
   total:values.reduce((n,v)=>n+v.capability_total_n+v.unwanted_generalization_total_n,0),
   kept:values.reduce((n,v)=>n+v.capability_retained_n+v.unwanted_generalization_retained_n,0),
   missing:values.reduce((n,v)=>n+v.capability_missing_primary_n+v.unwanted_generalization_missing_primary_n,0)};
  if($('cohort').value==='main'&&$('seed').value==='all')p.ci=data.cells.find(c=>c.variant===variant&&c.task_id===$('task').value&&c.model_family===$('model').value&&(c.method==='baseline'?'sft':c.method)===method);
  return p;
 });
}
function nondominated(p,all){return Number.isFinite(p.cap)&&Number.isFinite(p.ug)&&!all.some(q=>Number.isFinite(q.cap)&&Number.isFinite(q.ug)&&q.cap>=p.cap-1e-12&&q.ug<=p.ug+1e-12&&(q.cap>p.cap+1e-12||q.ug<p.ug-1e-12));}
function refresh(){
 clearError();points=computePoints();points.forEach(p=>p.front=nondominated(p,points));
 $('title').textContent=tasks[$('task').value]||$('task').value;
 $('subtitle').textContent=`${models[$('model').value]} · ${scopeRuns()[0]?.category||''} · ${eligibleRuns().length} selected runs`;
 $('protocol').textContent=data.protocols[$('variant').value];
 $('scope').textContent=$('cohort').value==='qwen32'?'Supplemental 32B cohort · one training seed · mixed A100/H200 inference. No 32B SFT or vanilla controls; medical advice has KL only.':'Main benchmark · all five trained seeds are available. Vanilla is a separate reference, not seed 0. Main and supplemental cohorts are never pooled.';
 const totalN=points.reduce((n,p)=>n+p.total,0), kept=points.reduce((n,p)=>n+p.kept,0),missing=points.reduce((n,p)=>n+p.missing,0);
 $('stats').replaceChildren(...[
  ['Selected runs',nf(eligibleRuns().length),`${points.length} methods / references`],
  ['Planned completions',nf(totalN),'Both evaluation axes'],
  ['Retained for scoring',pct(totalN?kept/totalN:null),`${nf(kept)} eligible completions`],
  ['Missing task scores',nf(missing),'Not treated as zero']
 ].map(([label,value,note])=>{const e=el('div',undefined,'stat');e.append(el('span',label),el('strong',value),el('p',note));return e;}));
 drawPlot();drawTable();
 $('front').replaceChildren(...points.filter(p=>p.front).map(p=>{const e=el('div',undefined,'front-item');e.append(dot(p.method),el('strong',names[p.method]));return e;}));
 if(!points.some(p=>p.front))$('front').append(el('p','No complete points in this selection.','muted'));
 const runs=eligibleRuns();options('run',runs.map(r=>r.id),Object.fromEntries(runs.map(r=>[r.id,runLabel(r)])));
 offset=0;activeId='';if(tab==='samples')loadSamples();saveState();
}
function svgEl(tag,attrs={},text){const e=document.createElementNS('http://www.w3.org/2000/svg',tag);for(const[k,v]of Object.entries(attrs))e.setAttribute(k,String(v));if(text!==undefined)e.textContent=text;return e;}
function drawPlot(){
 const svg=svgEl('svg',{viewBox:'0 0 660 380',class:'plot-svg',role:'img','aria-label':'Pareto plot of capability versus unwanted generalization'});
 svg.append(svgEl('rect',{x:68,y:20,width:560,height:300,fill:'#fbfcff'}));
 const x=v=>68+560*v,y=v=>320-300*v;
 for(let i=0;i<=5;i++){let v=i/5;svg.append(svgEl('line',{x1:x(v),y1:20,x2:x(v),y2:320,stroke:'#e5eaf1'}),svgEl('line',{x1:68,y1:y(v),x2:628,y2:y(v),stroke:'#e5eaf1'}),svgEl('text',{x:x(v),y:342,'text-anchor':'middle'},i*20+'%'),svgEl('text',{x:56,y:y(v)+4,'text-anchor':'end'},i*20+'%'));}
 svg.append(svgEl('text',{x:350,y:375,'text-anchor':'middle'},'Capability →'),svgEl('text',{transform:'translate(16 177) rotate(-90)','text-anchor':'middle'},'Unwanted generalization →'));
 const front=points.filter(p=>p.front).sort((a,b)=>a.cap-b.cap);
 if(front.length>1)svg.append(svgEl('polyline',{points:front.map(p=>`${x(p.cap)},${y(p.ug)}`).join(' '),fill:'none',stroke:'#75879f','stroke-width':1.5,'stroke-dasharray':'5 5'}));
 for(const p of points){if(p.cap===null||p.ug===null)continue;
  if(p.ci){const c=p.ci;const nums=['capability_ci_low','capability_ci_high','unwanted_generalization_ci_low','unwanted_generalization_ci_high'].map(k=>c[k]===''||c[k]==null?null:Number(c[k]));
   if(nums.every(Number.isFinite)){const[a,b,c,d]=nums;svg.append(svgEl('line',{x1:x(a),x2:x(b),y1:y(p.ug),y2:y(p.ug),stroke:colors[p.method],opacity:.65}),svgEl('line',{x1:x(p.cap),x2:x(p.cap),y1:y(c),y2:y(d),stroke:colors[p.method],opacity:.65}));}}
  const circle=svgEl('circle',{cx:x(p.cap),cy:y(p.ug),r:p.front?8:6,fill:colors[p.method],class:'point',tabindex:0,role:'button','aria-label':`${names[p.method]}: capability ${pct(p.cap)}, unwanted generalization ${pct(p.ug)}. Read samples.`});
  circle.append(svgEl('title',{},`${names[p.method]}\nCapability: ${pct(p.cap)}\nUnwanted gen.: ${pct(p.ug)}${p.front?'\nOn observed frontier':''}`));
  const go=()=>openRun(p.runs[0].id);circle.addEventListener('click',go);circle.addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();go();}});svg.append(circle);
 }
 $('plot').replaceChildren(svg);$('legend').replaceChildren(...points.map(p=>{const e=el('span');e.append(dot(p.method),el('span',names[p.method]));return e;}));
 $('plot-note').textContent='Click a point to read its samples. Dashed line: frontier among selected methods. '+($('cohort').value==='main'&&$('seed').value==='all'?'Bars: published 95% training-seed bootstrap intervals; vanilla has no interval.':'No uncertainty intervals estimated for single-seed selections.');
}
function drawTable(){
 $('summary').replaceChildren(...points.map(p=>{const tr=el('tr'),method=el('td');method.append(dot(p.method),el('span',names[p.method]));if(p.front)method.append(el('span',' · frontier','small'));tr.append(method);
  for(const s of [p.method==='vanilla'?'1 reference':p.runs.length+' seed'+(p.runs.length>1?'s':''),pct(p.cap),pct(p.ug),`${nf(p.kept)} / ${nf(p.total)}`,nf(p.missing)])tr.append(el('td',s));
  const cell=el('td'),b=el('button','Read samples');b.addEventListener('click',()=>openRun(p.runs[0].id));cell.append(b);tr.append(cell);return tr;}));
}
function showTab(name){tab=name;for(const t of ['overview','samples']){$(t).hidden=t!==name;$('tab-'+t).classList.toggle('active',t===name);$('tab-'+t).setAttribute('aria-selected',String(t===name));}if(name==='samples')loadSamples();saveState();}
function openRun(id){$('run').value=id;offset=0;activeId='';showTab('samples');}
function sampleQuery(){return {run:$('run').value,variant:$('variant').value,axis:$('axis').value,retention:$('retention').value,search:$('search').value,sort:$('sort').value,min:$('min').value||0,max:$('max').value||100,offset,limit:20};}
async function loadSamples(){
 const version=++requestVersion;++detailVersion;clearError();saveState();
 if(!$('run').value){$('sample-list').replaceChildren(el('p','Select at least one method.'));$('reader').replaceChildren();return;}
 $('sample-count').textContent='Loading samples…';$('reader').replaceChildren(el('p','Select a sample to read.','muted'));
 try{const d=await api('/api/samples',sampleQuery());if(version!==requestVersion)return;total=d.total;
  $('sample-count').textContent=`${nf(d.total)} matching samples / ${nf(d.run_total)} in run`;
  $('sample-states').textContent='Full run: '+Object.entries(d.states).map(([k,v])=>`${nf(v)} ${k.replaceAll('_',' ')}`).join(' · ');
  $('sample-list').replaceChildren(...d.rows.map(r=>{const b=el('button',undefined,'sample-button');b.dataset.id=r.id;const m=el('div',undefined,'meta');m.append(el('span',r.axis==='capability'?'CAPABILITY':'UNWANTED GEN.'),el('span',pct(r.value)));b.append(m,el('strong',r.question||'(No answer: inference failed)'),el('span',r.preview||'(No completion)','preview'),el('span',`Coherence ${r.coherence??'—'} · ${r.retention.replaceAll('_',' ')}`,'small muted'));b.addEventListener('click',()=>loadDetail(r.id));return b;}));
  if(!d.rows.length)$('sample-list').append(el('p','No samples match these filters.','panel muted'));
  $('page').textContent=d.total?`${offset+1}–${Math.min(offset+20,d.total)}`:'0';$('prev').disabled=offset===0;$('next').disabled=offset+20>=d.total;histogram(d.histogram);
  if(d.rows.length)loadDetail(d.rows.some(r=>r.id===activeId)?activeId:d.rows[0].id);
 }catch(e){if(version===requestVersion)error(e);}
}
function histogram(bins){const s=svgEl('svg',{viewBox:'0 0 250 67',role:'img','aria-label':'Full-run coherence distribution, bins 0 to 100'}),max=Math.max(...bins,1);bins.forEach((n,i)=>{const h=35*n/max,r=svgEl('rect',{x:5+i*24,y:40-h,width:18,height:h,fill:'#7297c5'});r.append(svgEl('title',{},`${i*10}–${i===9?100:i*10+9}: ${n} answers`));s.append(r);});s.append(svgEl('text',{x:5,y:60,'font-size':9,fill:'#66758a'},'0'),svgEl('text',{x:80,y:60,'font-size':9,fill:'#66758a'},'Coherence · full run'),svgEl('text',{x:228,y:60,'font-size':9,fill:'#66758a'},'100'));$('histogram').replaceChildren(s);}
function answerCard(row,title){const box=el('div',undefined,'answer-card');box.append(el('h3',title));const scores=el('div',undefined,'score-strip');scores.append(el('span',`${row.score_name||'Task'}: ${row.primary??'missing'}`),el('span',`Coherence: ${row.coherence??'missing'}`));box.append(scores);
 const answer=row.completion||'(No generated answer is available.)',reasoning=answer.match(/^<think>([\s\S]*?)<\/think>\s*([\s\S]*)$/);
 if(reasoning){const fold=el('details');fold.append(el('summary',reasoning[1].trim()?'Show reasoning (included in judging)':'Empty reasoning block'),el('pre',reasoning[1],'answer'));box.append(fold,el('pre',reasoning[2],'answer'));}else box.append(el('pre',answer,'answer'));
 const d=el('details'),summary=el('summary','Raw judge scores & labels');d.append(summary,el('pre',JSON.stringify(row.judgments,null,2),'raw'));box.append(d);return box;}
async function loadDetail(id,compare=''){
 const version=++detailVersion;activeId=id;saveState();document.querySelectorAll('.sample-button').forEach(b=>b.classList.toggle('active',b.dataset.id===id));
 try{const d=await api('/api/detail',{run:$('run').value,id,compare});if(version!==detailVersion)return;const r=d.sample;const reader=$('reader');reader.replaceChildren();
  const head=el('div',undefined,'panel-head');head.append(el('h2','Sample inspection'),el('span',r.axis==='capability'?'Capability':'Unwanted generalization','badge'));reader.append(head,el('p',r.question||'(Question unavailable for failed inference)','question'));
  const actions=el('div',undefined,'detail-actions'),copy=el('button','Copy answer'),exp=el('button','Export sample JSON');copy.onclick=()=>navigator.clipboard.writeText(r.completion).then(()=>toast('Answer copied')).catch(error);exp.onclick=()=>download('sample-'+r.id+'.json',JSON.stringify(d,null,2));actions.append(copy,exp);reader.append(actions);
  const label=el('label','Compare with another run'),s=el('select');s.id='compare';s.append(option('','No comparison'),...scopeRuns().filter(v=>v.id!==$('run').value).map(v=>option(v.id,runLabel(v))));s.value=compare;s.onchange=()=>loadDetail(id,s.value);label.append(s);reader.append(label);
  const answers=el('div',undefined,compare?'answers compare':'answers');answers.append(answerCard(r,runLabel(data.runs.find(v=>v.id===$('run').value))));
  if(compare){const box=el('div');if(d.comparisons.length){const l=el('label','Matching question · independent samples'),choose=el('select');d.comparisons.forEach((c,i)=>choose.append(option(String(i),`Answer ${i+1} · ${c.completion_id.slice(-26)}`)));const card=el('div');const render=()=>card.replaceChildren(answerCard(d.comparisons[Number(choose.value)],runLabel(data.runs.find(v=>v.id===compare))));choose.onchange=render;l.append(choose);box.append(l,card);render();}else box.append(el('p','No exact matching question and axis in this run.','muted'));answers.append(box);reader.append(el('p',d.matching,'small muted'));}reader.append(answers);
  const provenance=el('details'),sum=el('summary','Completion identity & provenance');provenance.append(sum,el('pre',JSON.stringify({completion_id:r.completion_id,eval_id:r.eval_id,coherence_source:r.coherence_source,run_id:d.run.id,source:d.run.source,training_job:d.run.training_job,revision:d.run.revision,archive:d.run.registry?.primary_result_artifact_id},null,2),'raw'));
  if(d.run.hf&&d.run.hf.startsWith('https://huggingface.co/')){const a=el('a','Hugging Face checkpoint');a.href=d.run.hf;a.target='_blank';a.rel='noopener noreferrer';provenance.append(a);}reader.append(provenance);
 }catch(e){if(version===detailVersion)error(e);}
}
function saveState(){if(!data)return;const q=new URLSearchParams();for(const key of ['cohort','task','model','variant','seed','run','axis','retention','sort','search','min','max'])q.set(key,$(key).value);q.set('tab',tab);q.set('methods',[...selectedMethods].join(','));history.replaceState(null,'','/?'+q);}
async function init(){try{data=await api('/api/catalog');const q=new URLSearchParams(location.search);
 for(const key of ['cohort','variant'])if(q.has(key)&&[...$(key).options].some(o=>o.value===q.get(key)))$(key).value=q.get(key);
 if(!data.runs.some(r=>r.cohort===$('cohort').value))$('cohort').value='main';
 syncOptions(true);for(const key of ['task','model'])if(q.has(key))$(key).value=q.get(key);if(!$('task').value)syncOptions(true);else syncOptions();
 if(q.has('methods')){selectedMethods=new Set(q.get('methods').split(',').filter(Boolean));document.querySelectorAll('.method-option').forEach(l=>{l.firstChild.checked=[...selectedMethods].some(m=>(names[m]||m)===l.lastChild.textContent);});}
 for(const key of ['seed','axis','retention','sort','search','min','max'])if(q.has(key))$(key).value=q.get(key);
 if(!$('seed').value)$('seed').value='all';refresh();if(q.has('run')&&[...$('run').options].some(o=>o.value===q.get('run')))$('run').value=q.get('run');
 $('definitions').replaceChildren(...data.caveats.map(s=>el('p',s)));for(const [k,v]of Object.entries(data.protocols))$('definitions').append(el('p',`${k}: ${v}`));
 if(data.notices.length)error(data.notices.join('\n'));if(q.get('tab')==='samples')showTab('samples');
 }catch(e){error(e);}}
for(const id of ['cohort','task','model'])$(id).addEventListener('change',()=>{syncOptions(id==='cohort');refresh();});
for(const id of ['variant','seed'])$(id).addEventListener('change',refresh);
for(const id of ['run','axis','retention','sort','min','max'])$(id).addEventListener('change',()=>{offset=0;activeId='';loadSamples();});
let searchTimer;$('search').addEventListener('input',()=>{clearTimeout(searchTimer);searchTimer=setTimeout(()=>{offset=0;loadSamples();},300);});
$('reset-samples').onclick=()=>{$('axis').value='all';$('retention').value='all';$('sort').value='original';$('search').value='';$('min').value='0';$('max').value='100';offset=0;loadSamples();};
$('prev').onclick=()=>{offset=Math.max(0,offset-20);loadSamples();};$('next').onclick=()=>{offset+=20;loadSamples();};
$('tab-overview').onclick=()=>showTab('overview');$('tab-samples').onclick=()=>showTab('samples');
$('share').onclick=()=>navigator.clipboard.writeText(location.href).then(()=>toast('View link copied (requires the local viewer)')).catch(error);
$('save-plot').onclick=()=>{const s=$('plot').firstChild.cloneNode(true);s.setAttribute('viewBox','0 -35 660 490');s.removeAttribute('class');s.append(svgEl('text',{x:68,y:-14,'font-size':16},`${tasks[$('task').value]} · ${models[$('model').value]}`),svgEl('text',{x:68,y:440,'font-size':10},`${$('variant').value} · ${$('seed').selectedOptions[0].textContent}`));s.querySelectorAll('text').forEach(t=>t.setAttribute('font-family','sans-serif'));points.forEach((p,i)=>{s.append(svgEl('circle',{cx:78+(i%3)*195,cy:395+Math.floor(i/3)*17,r:4,fill:colors[p.method]}),svgEl('text',{x:88+(i%3)*195,y:399+Math.floor(i/3)*17,'font-size':10,'font-family':'sans-serif'},names[p.method]));});download('pareto-'+$('task').value+'.svg',new XMLSerializer().serializeToString(s),'image/svg+xml');};
$('export-summary').onclick=()=>download('comparison-'+$('task').value+'.json',JSON.stringify({cohort:$('cohort').value,task:$('task').value,model:$('model').value,variant:$('variant').value,protocol:data.protocols[$('variant').value],points:points.map(({runs,...p})=>({...p,run_ids:runs.map(r=>r.id)}))},null,2));
$('help').onclick=()=>{$('methodology').open=true;$('methodology').scrollIntoView({behavior:'smooth'});};
init();
