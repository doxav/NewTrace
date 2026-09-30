/** Build only the completed P1 supplement from the verified presentation projection. */
import fs from 'node:fs/promises';
import path from 'node:path';
import crypto from 'node:crypto';
import assert from 'node:assert/strict';
import {fileURLToPath} from 'node:url';
import {Workbook, SpreadsheetFile} from '@oai/artifact-tool';

const HERE=path.dirname(fileURLToPath(import.meta.url));
const INVESTIGATION=path.resolve(HERE,'../..');
const REPO=path.resolve(INVESTIGATION,'../../..');
const OUTPUT=path.join(REPO,'outputs/01a07fdf-c364-79d1-bfe2-c467a140e0b7/optimizer_discovery_p1.xlsx');
const QA=path.join(HERE,'workbook_previews');
const hash=bytes=>crypto.createHash('sha256').update(bytes).digest('hex');
const inputHashes=new Map();
const checks=[];
const sources=[];
const readBytes=async filename=>{const bytes=await fs.readFile(filename);inputHashes.set(filename,hash(bytes));return bytes;};
const dataPath=path.join(HERE,'data.json');
const data=JSON.parse((await readBytes(dataPath)).toString('utf8'));
const sourceRow=(filename,section,digest,kind='SHA-256 des octets')=>[`S${String(sources.length+1).padStart(2,'0')}`,filename,section,digest,kind];
sources.push(sourceRow(path.relative(REPO,dataPath),'Projection numérique vérifiée',inputHashes.get(dataPath)));
for(const name of ['per_seed','contrasts','searches','usage']){const filename=path.join(HERE,`${name}.csv`);await readBytes(filename);sources.push(sourceRow(path.relative(REPO,filename),`Export tabulaire : ${name}`,inputHashes.get(filename)));}
for(const [relative,expected] of Object.entries(data.input_byte_hashes)){
  const filename=path.resolve(INVESTIGATION,relative);
  assert(filename.startsWith(INVESTIGATION+path.sep));
  await readBytes(filename);assert.equal(inputHashes.get(filename),expected,'Primary input hash changed');
  sources.push(sourceRow(path.relative(REPO,filename),'Source primaire de la projection',expected));
}

const seeds=[16411,16423,16437,16441,16453,16467];
const arms=['A0','I','C','R','W','B2'];
const genArms=['I','C','R','W'];
assert.equal(data.per_seed.length,36);assert.equal(data.contrasts.length,6);
assert.equal(data.searches.length,24);assert.equal(data.usage.length,4);
for(const seed of seeds) for(const arm of arms) assert.equal(data.per_seed.filter(r=>r.outer_seed===seed&&r.arm===arm).length,1);
for(const seed of seeds) for(const arm of genArms) assert.equal(data.searches.filter(r=>r.outer_seed===seed&&r.arm===arm).length,1);
assert(data.searches.every(r=>r.allocated_responses===8&&r.eligible_generated+r.ineligible_generated===8));
assert.equal(data.searches.reduce((n,r)=>n+r.allocated_responses,0),192);
assert.equal(data.searches.reduce((n,r)=>n+r.ineligible_generated,0),18);
assert.equal(data.per_seed.reduce((n,r)=>n+r.fallback_trajectories,0),0);
assert.equal(data.usage.reduce((n,r)=>n+r.responses,0),192);
assert.equal(data.usage.reduce((n,r)=>n+r.total_tokens,0),6980225);
const mean=values=>values.reduce((n,x)=>n+x,0)/values.length;
const median=values=>{const s=[...values].sort((a,b)=>a-b),n=s.length;return n%2?s[(n-1)/2]:(s[n/2-1]+s[n/2])/2;};
function close(actual,expected,label){assert(Number.isFinite(actual)&&Number.isFinite(expected),label);assert(Math.abs(actual-expected)<=1e-12*Math.max(1,Math.abs(expected)),`${label}: ${actual} != ${expected}`);}
const rows=[...data.per_seed].sort((a,b)=>seeds.indexOf(a.outer_seed)-seeds.indexOf(b.outer_seed)||arms.indexOf(a.arm)-arms.indexOf(b.arm));
const rowIndex=(seed,arm)=>6+rows.findIndex(r=>r.outer_seed===seed&&r.arm===arm);
for(const contrast of data.contrasts){const [first,second]=contrast.contrast.split('-');for(const seed of seeds){const a=rows.find(r=>r.outer_seed===seed&&r.arm===first),b=rows.find(r=>r.outer_seed===seed&&r.arm===second);close(a.auc-b.auc,contrast[`delta_${seed}`],`${contrast.contrast}/${seed}`);}close(mean(seeds.map(seed=>contrast[`delta_${seed}`])),contrast.mean,contrast.contrast);}

const wb=Workbook.create();
const names=['Données','Contrastes','Recherches','Usage','Sources'];
const sheets=Object.fromEntries(names.map(name=>[name,wb.worksheets.add(name)]));
const font='Liberation Sans';
function column(index){let n=index+1,s='';while(n){n--;s=String.fromCharCode(65+n%26)+s;n=Math.floor(n/26);}return s;}
function title(name,text,note,detail='AUC et regret : plus bas = mieux. Toutes les seeds, pertes et invalidités sont conservées.'){
  const sh=sheets[name];sh.showGridLines=false;
  sh.getRange('A1').values=[[text]];sh.getRange('A1').format.font={name:font,size:17,bold:true,color:'#263D50'};
  sh.getRange('A1:L1').format.rowHeight=30;
  sh.getRange('A2').values=[[note]];sh.getRange('A3').values=[[detail]];
  sh.getRange('A2:L3').format.font={name:font,size:10,italic:true,color:'#546472'};sh.getRange('A2:L3').format.rowHeight=21;
}
const sections=[];
function table(name,start,headers,values,{widths={},formats={},freeze=false}={}){
  const sh=sheets[name],end=start+values.length,last=column(headers.length-1);
  assert(values.every(row=>row.length===headers.length));
  const range=sh.getRange(`A${start}:${last}${end}`);range.values=[headers,...values];
  range.format.font={name:font,size:11,color:'#182D3D'};range.format.columnWidth=17;range.format.rowHeight=24;range.format.verticalAlignment='center';range.setNumberFormat('0.000000');
  const header=sh.getRange(`A${start}:${last}${start}`);header.format={fill:'#314A5E',font:{name:font,size:11,bold:true,color:'#FFFFFF'},rowHeight:48,wrapText:true,verticalAlignment:'center'};
  for(let r=start+1;r<=end;r++)if((r-start)%2===0)sh.getRange(`A${r}:${last}${r}`).format.fill='#F2F6F8';
  for(const [c,f] of Object.entries(formats))sh.getRange(`${column(+c)}${start+1}:${column(+c)}${end}`).setNumberFormat(f);
  for(const [c,w] of Object.entries(widths))sh.getRange(`${column(+c)}${start}:${column(+c)}${end}`).format.columnWidth=w;
  sh.tables.add(`A${start}:${last}${end}`,true,`${name.normalize('NFD').replace(/[\u0300-\u036f]/g,'')}${start}Table`);
  if(freeze)sh.freezePanes.freezeRows(5);
  sections.push({sheet:name,start,end,columns:headers.length});
}
function formula(name,cell,value,expected){sheets[name].getRange(cell).formulas=[[value]];if(expected!==undefined)checks.push({sheet:name,cell,formula:value,expected});}

title('Données','P1 et contrôle B2 : résultats par seed','Six seeds externes. A0/I/C/R/W : P1 exploratoire ; B2 : référence fixe supplémentaire.');
table('Données',5,['Seed externe','Bras','AUC audit','Regret final','Cible atteinte','Temps cible censuré','Traj. candidat invalides','Trajectoires de repli','Index sélectionné','SHA-256 source'],rows.map(r=>[r.outer_seed,r.arm,r.auc,r.final_regret,r.target_attainment,r.capped_target_evaluations,r.candidate_invalid_trajectories,r.fallback_trajectories,r.selection_index,r.source_sha256]),{widths:{0:17,1:12,5:25,6:27,7:26,8:23,9:75},formats:{0:'0',4:'0.00%',6:'0',7:'0',8:'0'},freeze:true});
table('Données',45,['Bras','AUC moyenne','AUC médiane','Regret final moyen','Cible atteinte','Traj. candidat invalides','Trajectoires de repli'],arms.map(arm=>[arm,null,null,null,null,null,null]),{widths:{0:17,1:21,2:21,3:27,4:21,5:28,6:28},formats:{4:'0.00%',5:'0',6:'0'}});
for(let i=0;i<arms.length;i++){const row=46+i,arm=arms[i],selected=rows.filter(r=>r.arm===arm);formula('Données',`B${row}`,`=AVERAGEIF(B6:B41,A${row},C6:C41)`,mean(selected.map(r=>r.auc)));formula('Données',`C${row}`,`=MEDIAN(${seeds.map(seed=>`C${rowIndex(seed,arm)}`).join(',')})`,median(selected.map(r=>r.auc)));formula('Données',`D${row}`,`=AVERAGEIF(B6:B41,A${row},D6:D41)`,mean(selected.map(r=>r.final_regret)));formula('Données',`E${row}`,`=AVERAGEIF(B6:B41,A${row},E6:E41)`,mean(selected.map(r=>r.target_attainment)));formula('Données',`F${row}`,`=SUMIF(B6:B41,A${row},G6:G41)`,selected.reduce((n,r)=>n+r.candidate_invalid_trajectories,0));formula('Données',`G${row}`,`=SUMIF(B6:B41,A${row},H6:H41)`,selected.reduce((n,r)=>n+r.fallback_trajectories,0));}

const interpretation={'positive signal':'signal positif','negative signal':'signal négatif','inconclusive':'inconclusif','no detectable difference':'pas de différence détectable'};
title('Contrastes','Contrastes appariés enregistrés','Delta = première politique moins seconde. Delta positif : première politique moins bonne.','Intervalles bootstrap importés inchangés. Six seeds ; lecture exploratoire, sans garantie de multiplicité.');
table('Contrastes',5,['Contraste','Portée','Moyenne ΔAUC','Médiane ΔAUC','IC 95 % bas','IC 95 % haut','Lecture enregistrée'],data.contrasts.map(r=>[r.contrast,r.scope==='primary_exploratory'?'P1 exploratoire':'B2 supplémentaire',null,null,r.ci_low,r.ci_high,interpretation[r.interpretation]??r.interpretation]),{widths:{0:17,1:27,2:24,3:24,4:21,5:21,6:27}});
table('Contrastes',15,['Seed externe',...data.contrasts.map(r=>r.contrast)],seeds.map(seed=>[seed,...data.contrasts.map(()=>null)]),{formats:{0:'0'},widths:{0:17}});
for(let i=0;i<data.contrasts.length;i++){const r=data.contrasts[i],rr=6+i,[a,b]=r.contrast.split('-'),c=column(i+1);for(let j=0;j<seeds.length;j++){const seed=seeds[j];formula('Contrastes',`${c}${16+j}`,`='Données'!C${rowIndex(seed,a)}-'Données'!C${rowIndex(seed,b)}`,r[`delta_${seed}`]);}formula('Contrastes',`C${rr}`,`=AVERAGE(${c}16:${c}21)`,r.mean);formula('Contrastes',`D${rr}`,`=MEDIAN(${c}16:${c}21)`,r.median);}
sheets.Contrastes.getRange('B16:G21').conditionalFormats.add('cellIs',{operator:'greaterThan',formula:0,format:{fill:'#FDEDE8'}});
sheets.Contrastes.getRange('B16:G21').conditionalFormats.add('cellIs',{operator:'lessThan',formula:0,format:{fill:'#EAF3F5'}});
sheets.Contrastes.getRange('A24').values=[['R−I reste central. R−C et W−R ont un ordre de précédence 5/1 ; une dérive temporelle reste possible.']];
sheets.Contrastes.getRange('A24').format.font={name:font,size:10,color:'#546472'};

title('Recherches','P1 : allocations et éligibilité des 24 pools','Chaque pool contient le seed et huit slots de génération alloués ; aucun slot invalide n’est remplacé.','Allocations de trajectoires incluant le seed. Les occasions inutilisées ne sont pas recyclées.');
const searches=[...data.searches].sort((a,b)=>seeds.indexOf(a.outer_seed)-seeds.indexOf(b.outer_seed)||genArms.indexOf(a.arm)-genArms.indexOf(b.arm));
table('Recherches',5,['Seed externe','Bras','Réponses allouées','Générés éligibles','Générés non éligibles','Fraction éligible','Seed sélectionné 0/1','Traj. train allouées','Traj. validation allouées'],searches.map(r=>[r.outer_seed,r.arm,r.allocated_responses,r.eligible_generated,r.ineligible_generated,null,r.selected_seed,r.train_allocations,r.validation_allocations]),{formats:{0:'0',2:'0',3:'0',4:'0',5:'0.00%',6:'0',7:'#,##0',8:'#,##0'},widths:{0:17,1:12,2:24,3:25,4:28,5:24,6:28,7:27,8:30},freeze:true});
for(let i=0;i<24;i++)formula('Recherches',`F${6+i}`,`=D${6+i}/C${6+i}`,searches[i].eligible_generated/8);
table('Recherches',33,['Bras','Réponses allouées','Générés éligibles','Générés non éligibles','Seed sélectionné','Traj. train allouées','Traj. validation allouées'],genArms.map(arm=>[arm,null,null,null,null,null,null]),{formats:{1:'0',2:'0',3:'0',4:'0',5:'#,##0',6:'#,##0'},widths:{0:17,1:24,2:25,3:28,4:25,5:27,6:30}});
for(let i=0;i<4;i++){const rr=34+i,arm=genArms[i],selected=searches.filter(r=>r.arm===arm);for(const [dest,src,key] of [['B','C','allocated_responses'],['C','D','eligible_generated'],['D','E','ineligible_generated'],['E','G','selected_seed'],['F','H','train_allocations'],['G','I','validation_allocations']])formula('Recherches',`${dest}${rr}`,`=SUMIF(B6:B29,A${rr},${src}6:${src}29)`,selected.reduce((n,r)=>n+r[key],0));}

title('Usage','P1 : tokens et coûts rapportés','192 réponses complétées. Les tokens de raisonnement sont décrits séparément, sans addition au total.','Coûts connus des réponses. Les tentatives transport incertaines restent dans l’audit primaire.');
table('Usage',5,['Bras','Réponses','Tokens prompt','Tokens complétion','Tokens raisonnement','Tokens totaux','Coût rapporté USD'],data.usage.map(r=>[r.arm,r.responses,r.prompt_tokens,r.completion_tokens,r.reasoning_tokens,r.total_tokens,r.cost_usd]),{formats:{1:'0',2:'#,##0',3:'#,##0',4:'#,##0',5:'#,##0',6:'0.000000000'},widths:{0:17,1:16,2:26,3:27,4:29,5:26,6:29}});
table('Usage',13,['Total','Réponses','Tokens prompt','Tokens complétion','Tokens raisonnement','Tokens totaux','Coût rapporté USD'],[['P1',null,null,null,null,null,null]],{formats:{1:'0',2:'#,##0',3:'#,##0',4:'#,##0',5:'#,##0',6:'0.000000000'},widths:{0:17,1:16,2:26,3:27,4:29,5:26,6:29}});
for(const [c,key] of [['B','responses'],['C','prompt_tokens'],['D','completion_tokens'],['E','reasoning_tokens'],['F','total_tokens'],['G','cost_usd']])formula('Usage',`${c}14`,`=SUM(${c}6:${c}9)`,data.usage.reduce((n,r)=>n+r[key],0));

title('Sources','Sources et définitions','La projection et les quatre sources primaires sont vérifiées par SHA-256 avant export.','Chemins relatifs au dépôt. Aucun code source évalué ni résultat primaire n’a été modifié.');
table('Sources',5,['ID','Fichier','Section','SHA-256 des octets','Type'],sources,{widths:{0:12,1:92,2:50,3:76,4:32}});
sheets.Sources.getRange(`B6:C${5+sources.length}`).format.wrapText=true;sheets.Sources.getRange(`A6:E${5+sources.length}`).format.rowHeight=44;
table('Sources',18,['Repère','Définition'],[
 ['A0','Seed manuscrit inchangé. Référence historique sans génération.'],['I','Huit générations indépendantes depuis le seed.'],['C','Réécriture de parents choisis selon le score train, sans résultats dans le prompt.'],['R','Même recherche que C, avec feedback de trajectoire riche.'],['W','Deux parents sur quatre tours ; largeur et nombre de tours changent ensemble.'],['B2','Référence fixe supplémentaire : premier point au centre des bornes.'],['AUC','Moyenne du regret normalisé sur B = 32, puis poids égaux des six strates.'],['Cible','Regret normalisé ≤ 0,01 ; temps non atteint représenté par B+1 uniquement dans la moyenne censurée.'],['Invalidité','L’éligibilité générée reste séparée du résultat de déploiement avec repli commun.'],['Portée','Six réplications externes. Aucune nouveauté algorithmique, profondeur ou amortissement établi.'],['Figure','Données exactes de production/presentation/paired_results.png et paired_results.pdf.'],['Cellule vide','Information non applicable ou absente, sans remplacement par une valeur zéro.'],
],{widths:{0:19,1:116}});
sheets.Sources.getRange('B19:B30').format.wrapText=true;sheets.Sources.getRange('A19:B30').format.rowHeight=35;
sheets['Données'].getRange('A6:B41').format.horizontalAlignment='center';
sheets.Recherches.getRange('A6:B29').format.horizontalAlignment='center';

for(const r of checks)close(sheets[r.sheet].getRange(r.cell).values[0][0],r.expected,`${r.sheet}!${r.cell}`);
await fs.mkdir(QA,{recursive:true});
const scan=await wb.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!',options:{useRegex:true,maxResults:40},summary:'Recherche d’erreurs de formules P1'});
await fs.writeFile(path.join(HERE,'p1_formula_error_scan.ndjson'),scan.ndjson);
const previews=[];
for(const [name,range,label] of [['Données','A1:I12','donnees'],['Données','A44:G52','moyennes'],['Données','J5:J12','sources_selectionnees'],['Contrastes','A1:G12','contrastes'],['Contrastes','A14:G24','deltas'],['Recherches','A1:I12','recherches'],['Recherches','A32:G38','recherches_resume'],['Usage','A1:G15','usage'],['Sources','A1:E15','sources'],['Sources','A17:B31','definitions']]){const png=await wb.render({sheetName:name,range,scale:1.3,format:'png'});const filename=path.join(QA,`${label}.png`);await fs.writeFile(filename,new Uint8Array(await png.arrayBuffer()));previews.push({sheet:name,range,path:filename});}
for(const name of names){const result=await wb.inspect({kind:'table',range:`${name}!A5:D8`,include:'values,formulas',tableMaxRows:4,tableMaxCols:4,maxChars:1500});await fs.writeFile(path.join(HERE,`p1_inspect_${name}.ndjson`),result.ndjson);}
for(const [filename,digest] of inputHashes)assert.equal(hash(await fs.readFile(filename)),digest,'Input changed during export');
await fs.mkdir(path.dirname(OUTPUT),{recursive:true});await (await SpreadsheetFile.exportXlsx(wb)).save(OUTPUT);
const result={output:OUTPUT,sha256:hash(await fs.readFile(OUTPUT)),sheets:names,input_hashes:Object.fromEntries(inputHashes),per_seed_rows:36,contrast_rows:6,pools:24,allocated_responses:192,usage_rows:4,ineligible_generated:18,all_six_seeds_retained:true,formula_checks:checks,previews,sections,source_rows:sources.length,new_model_calls:0,new_objective_calls:0,old_workbook_touched:false};
await fs.writeFile(path.join(HERE,'p1_workbook_checks.json'),JSON.stringify(result,null,2)+'\n');
console.log(JSON.stringify({output:OUTPUT,sha256:result.sha256,sheets:names.length,verified_formulas:checks.length,previews:previews.length,sources:sources.length}));
