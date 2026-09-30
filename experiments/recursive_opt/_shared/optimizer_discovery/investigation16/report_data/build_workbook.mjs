/** Build the completed-study data workbook without importing experiment executors. */
import fs from 'node:fs/promises';
import path from 'node:path';
import zlib from 'node:zlib';
import crypto from 'node:crypto';
import assert from 'node:assert/strict';
import {fileURLToPath} from 'node:url';
import {Workbook, SpreadsheetFile} from '@oai/artifact-tool';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const INVESTIGATION = path.dirname(HERE);
const BASE = path.dirname(INVESTIGATION);
const REPO = path.resolve(BASE, '../..');
const OUTPUT = path.join(REPO, 'outputs/01a07fdf-c364-79d1-bfe2-c467a140e0b7/optimizer_discovery_data.xlsx');
const PREVIEWS = path.join(HERE, 'previews');
const sourceRows = [];
const inputHashes = new Map();
const sheetChecks = [];
const numericChecks = [];
const sections = [];
/** @param {number[]} values Nonempty observations. @returns {number} Arithmetic mean. */
const mean = values => values.reduce((sum, value) => sum + value, 0) / values.length;
/** @param {Uint8Array} bytes Exact file bytes. @returns {string} SHA-256 digest. */
const sha = bytes => crypto.createHash('sha256').update(bytes).digest('hex');
const allowed = ['statistics', 'generation', 'benchmark', 'selection', 'throughput', 'feedback_experiment'];

/** Read only an explicitly permitted completed-study source and track exact bytes.
 * @param {string} relative Source relative to the optimizer-discovery directory.
 * @param {string} section Consumed source fields.
 * @param {string} purpose Scientific provenance description.
 * @returns {Promise<{id: string, data: object}>} Source identifier and parsed record.
 */
async function read(relative, section, purpose) {
  const logical = path.resolve(BASE, relative);
  assert(logical === path.join(BASE, 'exp15_results.json') || allowed.some(name => logical.startsWith(path.join(INVESTIGATION, name) + path.sep)), 'Source outside completed-study allowlist');
  let physical = logical;
  try { await fs.access(physical); } catch { physical += '.gz'; }
  const bytes = await fs.readFile(physical);
  const digest = sha(bytes);
  if (inputHashes.has(physical)) assert.equal(inputHashes.get(physical), digest, 'Input changed during build');
  inputHashes.set(physical, digest);
  const id = `S${String(sourceRows.length + 1).padStart(3, '0')}`;
  sourceRows.push([id, path.relative(REPO, logical), section, digest, physical.endsWith('.gz') ? 'SHA-256 du fichier gzip' : 'SHA-256 des octets', purpose]);
  return {id, data: JSON.parse((physical.endsWith('.gz') ? zlib.gunzipSync(bytes) : bytes).toString('utf8'))};
}

/** Compare numbers at the precision required to preserve recorded scientific values.
 * @param {string} name Check identity.
 * @param {number} actual Recomputed value.
 * @param {number} expected Preserved value.
 * @param {number} tolerance Scaled numerical tolerance.
 * @returns {void}
 */
function close(name, actual, expected, tolerance=1e-12) {
  assert(Number.isFinite(actual) && Number.isFinite(expected));
  assert(Math.abs(actual - expected) <= tolerance * Math.max(1, Math.abs(expected)), `${name}: ${actual} != ${expected}`);
  numericChecks.push({name, actual, expected, tolerance});
}

const exp = await read('exp15_results.json', 'per_seed, arms, contrasts, accounting', 'Résultats confirmatoires EXP-15 inchangés');
const s0 = await read('investigation16/statistics/diagnostics.json', 'responses, feedback, selections, stratum_means, contrasts', 'Audit rétrospectif S0, aucune nouvelle génération');
const g1 = await read('investigation16/generation/analysis_results.json', 'rows, caps, usage_total', 'Audit final G1, tous les slots');
const b1 = await read('investigation16/benchmark/results.json', 'groups, mean_curve', 'Données exactes de headroom.png et headroom.pdf');
const b2 = await read('investigation16/benchmark/b2/results.json', 'paired_rows, groups', 'Données exactes des quatre panneaux paired_outcomes');
const b2freeze = await read('investigation16/benchmark/b2/freeze.json', 'allocations, source hashes', 'Identités des 96 paires B2');
const b2check = await read('investigation16/benchmark/b2/paired_outcomes_checks.json', 'panels, raw_input_byte_hashes', 'Vérification indépendante des données de la figure B2');
const s1 = await read('investigation16/selection/results.json', 'summaries, audit_bank_auc, accounting', 'Données exactes de selection_quality.png et .svg');
const s1freeze = await read('investigation16/selection/freeze.json', 'bank', 'Banque fixe, provenance et sources');
const t1 = await read('investigation16/throughput/results.json', 'timings', 'Mesures originales, y compris la mesure affectée par suspension');
const t1r = await read('investigation16/throughput/timing_repair_r1/results.json', 'corrected_summaries', 'Mesure de remplacement et moyennes corrigées');
const f1 = await read('investigation16/feedback_experiment/analysis_results.json', 'rows, conditions, contrasts, usage, total_allocation', 'Audit final F1, estimand de déploiement avec repli commun');

assert.equal(exp.data.per_seed.length, 5);
assert.equal(s0.data.responses.length, 80);
assert.equal(g1.data.rows.length, 24);
assert.equal(b1.data.total_trajectories, 384);
assert.equal(b2.data.paired_rows.length, 96);
assert.equal(s1.data.summaries.length, 9);
assert.equal(s1freeze.data.bank.length, 11);
assert.equal(f1.data.rows.length, 24);
assert.equal(g1.data.caps['8000'].eligible, 8);
assert.equal(g1.data.caps['32000'].eligible, 10);
assert.equal(f1.data.rows.filter(row => !row.train_valid).length, 4);
assert.equal(f1.data.rows.reduce((n,row) => n+row.transport_attempts,0),26);
assert.equal(f1.data.rows.reduce((n,row) => n+row.fallback_trajectories,0),24);
close('EXP15 A2-A1', exp.data.contrasts['A2-A1'].mean, -0.005067768508320613);
close('F1 rich-sparse',f1.data.contrasts['anytime_rich-anytime_sparse'].mean,0.0492286056750564);
close('F1 IC inférieur rich-sparse',f1.data.contrasts['anytime_rich-anytime_sparse'].paired_bootstrap_95[0],0.00016007339371492226);
close('F1 tokens',f1.data.rows.reduce((n,row)=>n+row.usage.total_tokens,0),321691,0);

const wb = Workbook.create();
const names = ['Guide','EXP15','EXP15_generations','S0_trace','S0_selection','S0_strates','G1','B1','B1_courbes','B2','S1_panels','S1_banque','T1','F1','F1_contrastes','F1_usage','Sources'];
const sheets = Object.fromEntries(names.map(name=>[name,wb.worksheets.add(name)]));
const font = 'Liberation Sans';

/** Convert a zero-based column index to Excel's stable A1 column identifier.
 * @param {number} index Zero-based column.
 * @returns {string} Excel column label.
 */
function col(index) { let value=index+1, out=''; while(value){value--;out=String.fromCharCode(65+value%26)+out;value=Math.floor(value/26);}return out; }

/** Initialize an unmerged, compact scientific data sheet.
 * @param {string} name Worksheet name.
 * @param {string} heading Main title.
 * @param {string} note Scope statement.
 * @param {string} subnote Interpretation convention.
 * @returns {void}
 */
function title(name, heading, note, subnote='AUC et regret : plus bas = mieux. Cellule vide : valeur absente, jamais un zéro imputé.') {
  const sh=sheets[name]; sh.showGridLines=false;
  sh.getRange('A1').values=[[heading]];
  sh.getRange('A1').format.font={name:font,size:17,bold:true,color:'#243747'};
  sh.getRange('A1:L1').format.rowHeight=29;
  sh.getRange('A2').values=[[note]];
  sh.getRange('A3').values=[[subnote]];
  sh.getRange('A2:L3').format.font={name:font,size:10,italic:true,color:'#596773'};
  sh.getRange('A2:L3').format.rowHeight=21;
}

/** Add one flat table; all numeric values stay typed and missing values stay blank.
 * @param {string} name Worksheet name.
 * @param {number} start Header row.
 * @param {string[]} headers Column labels.
 * @param {Array<Array<string|number|boolean|null>>} rows Source data.
 * @param {{formats?: Object<string, string>, widths?: Object<string, number>, label?: string, freeze?: boolean}} options Table presentation.
 * @returns {{start: number, end: number, headers: string[]}} Table bounds.
 */
function table(name, start, headers, rows, {formats={}, widths={}, label='Données', freeze=true}={}) {
  assert(rows.length>0 && rows.every(row=>row.length===headers.length));
  const sh=sheets[name], last=start+rows.length;
  const range=sh.getRange(`A${start}:${col(headers.length-1)}${last}`);
  range.values=[headers,...rows.map(row=>row.map(value=>typeof value === "boolean" ? Number(value) : value))];
  range.format.font={name:font,size:11,color:'#182A37'};
  range.format.rowHeight=22;
  range.format.verticalAlignment='center';
  range.format.columnWidth=15;
  range.setNumberFormat('0.000000');
  headers.forEach((_,index)=>{if(rows.some(row=>typeof row[index] === 'boolean'))sh.getRange(`${col(index)}${start+1}:${col(index)}${last}`).setNumberFormat('0');});
  const hr=sh.getRange(`A${start}:${col(headers.length-1)}${start}`);
  hr.format={fill:'#314A5E',font:{name:font,size:11,bold:true,color:'#FFFFFF'},wrapText:true,rowHeight:44,verticalAlignment:'center'};
  for(let row=start+1;row<=last;row++) if((row-start)%2===0) sh.getRange(`A${row}:${col(headers.length-1)}${row}`).format.fill='#F3F6F8';
  for(const [index,format] of Object.entries(formats)) sh.getRange(`${col(+index)}${start+1}:${col(+index)}${last}`).setNumberFormat(format);
  for(const [index,width] of Object.entries(widths)) sh.getRange(`${col(+index)}${start}:${col(+index)}${last}`).format.columnWidth=width;
  sh.tables.add(`A${start}:${col(headers.length-1)}${last}`,true,`${name.replaceAll('_','')}${start}Table`);
  if(freeze && last>18) sh.freezePanes.freezeRows(5);
  sections.push({sheet:name,label,start,end:last,columns:headers.length});
  return {start:start+1,end:last,headers};
}

/** Write one formula and independently compare its numeric result after calculation.
 * @param {string} name Worksheet name.
 * @param {string} address Destination cell.
 * @param {string} expression Excel formula.
 * @param {number} [expected] Independently computed result when available.
 * @returns {void}
 */
function formula(name, address, expression, expected) {
  sheets[name].getRange(address).formulas=[[expression]];
  if(expected!==undefined) sheetChecks.push({sheet:name,address,expected});
}

title('Guide','Données de recherche sur la découverte d’optimiseurs','EXP-15 et diagnostics terminés d’EXP-16. P1 en cours exclu de ce classeur.','Source : résultats préservés. Les intervalles sont importés du protocole, sans recalcul statistique dans Excel.');
table('Guide',5,['Étude','Onglets','Portée','Figure ou tableau'],[
 ['EXP-15 / S0','EXP15, EXP15_generations, S0_*','Confirmation à 5 seeds ; audit rétrospectif distinct','AUC, contrastes, invalidité, sélection et trace'],
 ['G1','G1','24 réponses ; faisabilité de génération, pas supériorité','Plafonds de 8 000 et 32 000 tokens'],
 ['B1','B1, B1_courbes','384 trajectoires de quatre politiques fixes','benchmark/headroom.png et .pdf'],
 ['B2','B2','96 paires publiques ; initialisation seulement','benchmark/b2/paired_outcomes.png et .pdf'],
 ['S1','S1_panels, S1_banque','200 sous-panels chevauchants ; banque fixe','selection/selection_quality.png et .svg'],
 ['T1','T1','Débit matériel ; mesure suspendue conservée','throughput/REPORT_T1.md, timings corrigés'],
 ['F1','F1, F1_contrastes, F1_usage','6 blocs à parent fixe ; déploiement avec repli','feedback_experiment/REPORT.md'],
 ['Provenance','Sources','Fichiers relatifs au dépôt et SHA-256 des octets','Renvoi par identifiant Sxxx dans les tables'],
],{widths:{0:19,1:38,2:60,3:53},freeze:false});
table('Guide',17,['Convention','Définition'],[
 ['Delta','Première politique moins la seconde. Delta AUC négatif : première meilleure.'],
 ['Valeur manquante','Cellule vide et statut explicite. Une invalidité ne reçoit pas de regret artificiel.'],
 ['Indicateurs','Les booléens de validité et de conservation sont codés 1 = oui, 0 = non. Ce codage ne concerne pas les regrets.'],
 ['Fallback / repli','Politique seed commune poursuivant la même trajectoire, sans remise à zéro du budget.'],
 ['Inférence','Bootstrap importé, fragile à 5 ou 6 blocs. Aucun gain futur promis.'],
 ['P1','Aucune réponse, trajectoire, sélection ou métrique P1 chargée. Aucune valeur provisoire n’est créée.'],
 ['Source exacte','Les chiffres des figures sont conservés sans écrêtage, même en cas de perte.'],
],{widths:{0:25,1:58},freeze:false});

title('EXP15','EXP-15 : résultats par seed','Holdout, 5 seeds externes. Déploiement avec repli commun.');
const expRows=[];
for(const block of exp.data.per_seed) for(const arm of ['A0','A1','A2']) {const r=block[arm];expRows.push([block.outer_seed,arm,r.auc,r.final_regret,r.target_attainment,r.capped_target_evaluations,r.fallback_trajectories,r.candidate_valid_trajectories,exp.id]);}
table('EXP15',5,['Seed externe','Bras','AUC','Regret final','Cible atteinte','Temps cible censuré','Replis','Trajectoires valides','Source'],expRows,{formats:{0:'0',4:'0.00%',6:'0',7:'0'},widths:{0:16,1:12,5:22,7:22}});
const contrastRows=Object.entries(exp.data.contrasts).map(([name,r])=>[name,r.mean,r.median,...r.paired_bootstrap_95,s0.data.contrasts[name].sample_sd,s0.data.contrasts[name].standard_error,r.interpretation,exp.id,s0.id]);
table('EXP15',24,['Contraste','Moyenne ΔAUC','Médiane ΔAUC','IC 95 % bas','IC 95 % haut','Écart-type apparié','Erreur standard','Lecture enregistrée','Source primaire','Source S0'],contrastRows,{widths:{0:18,7:23}});
table('EXP15',30,['Bras','AUC moyenne','AUC médiane','Contribution AUC t≤8','Part AUC t≤8','Source'],Object.entries(exp.data.arms).map(([arm,r])=>[arm,null,null,s0.data.early_auc_contributions[arm]['8'],null,s0.id]),{formats:{4:'0.00%'},widths:{0:18,3:25}});
for(let i=0;i<3;i++){let rr=31+i,arm=['A0','A1','A2'][i];formula('EXP15',`B${rr}`,`=AVERAGEIF(B6:B20,A${rr},C6:C20)`,exp.data.arms[arm].mean_auc);formula('EXP15',`C${rr}`,`=MEDIAN(C${6+i},C${9+i},C${12+i},C${15+i},C${18+i})`,exp.data.arms[arm].median_auc);formula('EXP15',`E${rr}`,`=D${rr}/B${rr}`);}

title('EXP15_generations','EXP-15 : toutes les propositions','80 réponses conservées. L’invalidité, les fins length et les copies restent présentes.');
table('EXP15_generations',5,['Seed externe','Bras','Slot','Éligible','Statut source','Fin','Fournisseur','Tokens prompt','Tokens complétion','Tokens raisonnement','Coût USD','Octets source','SHA-256 source','Source'],s0.data.responses.map(r=>[r.outer_seed,r.arm,r.slot,r.eligible,r.source_status,r.finish_reason,r.provider,r.prompt_tokens,r.completion_tokens,r.reasoning_tokens,r.cost_usd,r.source_bytes,r.source_sha256,s0.id]),{formats:{0:'0',2:'0',7:'#,##0',8:'#,##0',9:'#,##0',10:'0.000000000',11:'#,##0'},widths:{4:23,6:23,12:72}});

title('S0_trace','S0 : information visible dans les traces','40 prompts A2. Incumbent absent = point du meilleur résultat absent des observations montrées.');
const traceRows=s0.data.feedback.map(r=>{const p=r.panels.current;return[r.outer_seed,r.slot,r.json_valid,r.characters,p.tasks,p.observations,p.actual_incumbent_missing_from_shown_observations,p.raw_best_value_max_min_ratio,s0.id];});
table('S0_trace',5,['Seed externe','Slot','JSON valide','Caractères','Tâches','Observations montrées','Incumbents absents','Ratio valeurs brutes','Source'],traceRows,{formats:{0:'0',1:'0',3:'#,##0',4:'0',5:'0',6:'0',7:'0.000E+00'},widths:{5:25,6:24,7:24}});
table('S0_trace',49,['Tâches résumées','Incumbents absents','Fraction absente','Source'],[[null,null,null,s0.id]],{formats:{0:'0',1:'0',2:'0.00%'},widths:{0:24,1:24,2:24}});
formula('S0_trace','A50','=SUM(E6:E45)',240);formula('S0_trace','B50','=SUM(G6:G45)',167);formula('S0_trace','C50','=B50/A50');

title('S0_selection','S0 : sélection train et validation','10 pools EXP-15. Audit rétrospectif, pas une nouvelle règle de sélection.');
table('S0_selection',5,['Seed externe','Bras','Éligibles seed inclus','Index sélectionné','Meilleur index train','AUC train sélection','AUC validation','Rang train sélection','Spearman train-val','Changements retrait strate','Source'],s0.data.selections.map(r=>[r.outer_seed,r.arm,r.eligible_count_including_seed,r.selected_index,r.train_winner_index,r.selected_train_auc,r.selected_validation_auc,r.selected_train_rank,r.train_validation_spearman,r.leave_one_stratum_out_changes,s0.id]),{formats:{0:'0',2:'0',3:'0',4:'0',7:'0',9:'0'},widths:{2:25,5:23,7:24,8:25,9:28}});

title('S0_strates','S0 : résultats par strate','Moyennes descriptives EXP-15. Les strates ne sont pas des réplications externes indépendantes.');
table('S0_strates',5,['Strate','A0 AUC','A1 AUC','A2 AUC','A2−A1','Source'],Object.entries(s0.data.stratum_means).map(([name,r])=>[name,r.A0,r.A1,r.A2,null,s0.id]),{widths:{0:24}});
Object.entries(s0.data.stratum_means).forEach(([name,r],i)=>formula('S0_strates',`E${6+i}`,`=D${6+i}-C${6+i}`,r['A2-A1']));

title('G1','G1 : limites de génération','12 réponses par plafond. Les AUC manquantes restent vides pour les candidats invalides.');
table('G1',5,['Bloc','Contexte','Plafond tokens','Éligible','Statut source','Fin','AUC si valide','Tokens prompt','Tokens complétion','Tokens totaux','Coût USD','Durée réponse s','Tentatives','Fournisseur','SHA-256 source','Source'],g1.data.rows.map(r=>[r.block,r.context,r.cap,r.eligible,r.source_status,r.finish_reason,r.auc,r.usage.prompt_tokens,r.usage.completion_tokens,r.usage.total_tokens,r.usage.cost_usd,r.wall_s,r.transport_attempts,r.receipt?.provider_name??null,r.source_sha256,g1.id]),{formats:{0:'0',2:'#,##0',7:'#,##0',8:'#,##0',9:'#,##0',10:'0.000000000',11:'0.000',12:'0'},widths:{4:24,13:24,14:72}});
table('G1',33,['Plafond tokens','Réponses','Éligibles','Fraction éligible','Fins length','Coût USD','Tokens totaux','Source'],[8000,32000].map(cap=>[cap,null,null,null,null,null,null,g1.id]),{formats:{0:'#,##0',1:'0',2:'0',3:'0.00%',4:'0',5:'0.000000000',6:'#,##0'}});
for(let i=0;i<2;i++){const rr=34+i,cap=[8000,32000][i],v=g1.data.caps[cap];formula('G1',`B${rr}`,`=COUNTIF(C6:C29,A${rr})`,v.responses);formula('G1',`C${rr}`,`=COUNTIFS(C6:C29,A${rr},D6:D29,1)`,v.eligible);formula('G1',`D${rr}`,`=C${rr}/B${rr}`,v.eligible_fraction);formula('G1',`E${rr}`,`=COUNTIFS(C6:C29,A${rr},F6:F29,"length")`,v.length);formula('G1',`F${rr}`,`=SUMIF(C6:C29,A${rr},K6:K29)`,v.usage.cost_usd.reported_sum);formula('G1',`G${rr}`,`=SUMIF(C6:C29,A${rr},J6:J29)`,v.usage.total_tokens.reported_sum);}

title('B1','B1 : politiques fixes et surface optimisable','384 trajectoires. Données de la figure headroom ; aucun bras de génération comparé.');
const b1rows=[],curves=[];
for(const [condition,policies] of Object.entries(b1.data.groups)) for(const [policy,r] of Object.entries(policies)){const m=r.metrics;b1rows.push([condition,policy,r.n,r.valid,r.invalid,m.auc,m.final_regret,m.attained,m.capped_target_evaluations,m.early_mass_8,b1.id]);m.mean_curve.forEach((v,i)=>curves.push([condition,policy,i+1,v,b1.id]));close(`B1 courbe ${condition}/${policy}`,mean(m.mean_curve),m.auc);}
table('B1',5,['Distribution','Politique','Trajectoires','Valides','Invalides','AUC','Regret final','Cible atteinte','Temps cible censuré','Part AUC t≤8','Source'],b1rows,{formats:{2:'0',3:'0',4:'0',7:'0.00%',9:'0.00%'},widths:{0:19,1:25,8:25}});
title('B1_courbes','B1 : courbes moyennes exactes','32 points × 4 politiques × 2 distributions, sans écrêtage.');
table('B1_courbes',5,['Distribution','Politique','Évaluation t','Regret normalisé moyen','Source'],curves,{formats:{2:'0'},widths:{0:20,1:26,3:30}});

title('B2','B2 : les 96 comparaisons appariées','Premier point au centre. Données exactes des quatre scatterplots ; toutes les pertes conservées.');
const allocs=Object.fromEntries(b2freeze.data.allocations.map(r=>[r.id,r]));
const b2rows=[];
for(const pair of b2.data.paired_rows){
  const allocation=allocs[pair.id],c=pair.comparison,sm=c.seed.metrics,vm=c.variant.metrics;
  const sourceIDs=[];
  for(const [kind,rawRelative] of [['seed',`investigation16/benchmark/${allocation.control_path}`],['variant',`investigation16/benchmark/b2/raw/${pair.id}.json`]]){
    const source=await read(rawRelative,'metrics.auc, metrics.final_regret',`Paire B2 ${pair.id}, ${kind}`);
    const physical=path.join(BASE,rawRelative);
    assert.equal(inputHashes.get(physical),b2check.data.raw_input_byte_hashes[physical]);
    close(`B2 ${pair.id} ${kind} AUC`,source.data.metrics.auc,c[kind].metrics.auc);
    close(`B2 ${pair.id} ${kind} final`,source.data.metrics.final_regret,c[kind].metrics.final_regret);
    assert(source.data.valid && source.data.candidate_valid && !source.data.fallback_used);
    sourceIDs.push(source.id);
  }
  b2rows.push([pair.id,allocation.condition,`${allocation.task.family}/${allocation.task.dimension}`,allocation.local_seed,sm.auc,vm.auc,null,sm.final_regret,vm.final_regret,null,null,null,c.seed.valid===1,c.variant.valid===1,...sourceIDs,pair.control_sha256,pair.variant_sha256]);
}
table('B2',5,['Paire','Distribution','Strate','Seed local','AUC seed','AUC B2','Delta AUC','Regret final seed','Regret final B2','Delta regret final','Issue AUC','Issue finale','Seed valide','B2 valide','Source seed','Source B2','Hash canonique raw seed','Hash canonique raw B2'],b2rows,{formats:{3:'0'},widths:{0:24,1:20,2:21,7:24,8:24,9:24,16:72,17:72}});
for(let i=0;i<96;i++){const rr=6+i,c=b2.data.paired_rows[i].comparison;formula('B2',`G${rr}`,`=F${rr}-E${rr}`,c.delta.auc);formula('B2',`J${rr}`,`=I${rr}-H${rr}`,c.delta.final_regret);formula('B2',`K${rr}`,`=IF(G${rr}<0,"gain",IF(G${rr}>0,"perte","égalité"))`);formula('B2',`L${rr}`,`=IF(J${rr}<0,"gain",IF(J${rr}>0,"perte","égalité"))`);}
table('B2',106,['Distribution','AUC seed moyenne','AUC B2 moyenne','Réduction AUC moyenne','Gains AUC','Pertes AUC','Gains finaux','Pertes finales','Égalités finales','Source'],['central','broad'].map(c=>[c,null,null,null,null,null,null,null,null,b2.id]),{formats:{3:'0.00%',4:'0',5:'0',6:'0',7:'0',8:'0'},widths:{1:25,2:25,3:27}});
for(let i=0;i<2;i++){const rr=107+i,c=['central','broad'][i],saved=b2check.data.panels[c];formula('B2',`B${rr}`,`=AVERAGEIF(B6:B101,A${rr},E6:E101)`,saved.auc.means.seed);formula('B2',`C${rr}`,`=AVERAGEIF(B6:B101,A${rr},F6:F101)`,saved.auc.means.variant);formula('B2',`D${rr}`,`=1-C${rr}/B${rr}`);for(const [cc,metric,issue,expected] of [['E','K','gain',saved.auc.counts.gain],['F','K','perte',saved.auc.counts.loss],['G','L','gain',saved.final_regret.counts.gain],['H','L','perte',saved.final_regret.counts.loss],['I','L','égalité',saved.final_regret.counts.tie]])formula('B2',`${cc}${rr}`,`=COUNTIFS(B6:B101,A${rr},${metric}6:${metric}101,"${issue}")`,expected);}
assert.equal(b2.data.paired_rows.filter(r=>r.comparison.delta.auc>0).length,26);

title('S1_panels','S1 : précision de sélection dans la banque fixe','200 sous-panels chevauchants par configuration. Ce ne sont pas 200 réplications indépendantes.');
table('S1_panels',5,['Instances par strate','Seeds locaux','Instances totales','Trajectoires par candidat','Sous-panels','AUC audit moyenne','AUC audit médiane','Excès vs meilleur banque','Corrélation rang moyenne','Réduction vs 6×1','Source'],s1.data.summaries.map(r=>[r.instances_per_stratum,r.local_seeds,null,null,r.paired_draw_audit_values.length,r.selection_mean_audit_auc,r.selection_median_audit_auc,r.mean_excess_above_best_finite_bank_audit,r.mean_rank_correlation_to_audit,null,s1.id]),{formats:{0:'0',1:'0',2:'0',3:'0',4:'0',9:'0.00%'},widths:{0:25,3:30,5:25,6:25,7:30,8:30}});
for(let i=0;i<9;i++){const rr=6+i,r=s1.data.summaries[i];assert.equal(r.paired_draw_audit_values.length,200);assert.equal(Object.values(r.selection_frequency).reduce((a,b)=>a+b,0),200);close(`S1 moyenne ${i}`,mean(r.paired_draw_audit_values),r.selection_mean_audit_auc);formula('S1_panels',`C${rr}`,`=A${rr}*6`);formula('S1_panels',`D${rr}`,`=C${rr}*B${rr}`);formula('S1_panels',`J${rr}`,`=1-F${rr}/$F$6`);}
title('S1_banque','S1 : banque et fréquences de sélection','11 sources fixées avant mesure. Fréquences sur 200 sous-panels par configuration.');
table('S1_banque',5,['Index banque','Provenance','AUC audit',...s1.data.summaries.map(r=>`${r.instances_per_stratum*6}×${r.local_seeds}`),'SHA-256 source','Source banque','Source résultat'],s1freeze.data.bank.map(r=>[r.bank_index,r.provenance.join(' ; '),s1.data.audit_bank_auc[r.bank_index],...s1.data.summaries.map(s=>s.selection_frequency[String(r.bank_index)]),r.source_sha256,s1freeze.id,s1.id]),{formats:{0:'0',...Object.fromEntries(Array.from({length:9},(_,i)=>[i+3,'0']))},widths:{0:17,1:73,12:72}});
close('S1 24x2',s1.data.summaries.find(r=>r.instances_per_stratum===4&&r.local_seeds===2).selection_mean_audit_auc,0.07556555890500527);

title('T1','T1 : débit et correction de la suspension','La mesure originale affectée reste visible. Les moyennes corrigées l’excluent.','Durées monotones en secondes pour 24 trajectoires. Même science à chaque niveau de parallélisme.');
const timingRows=t1.data.timings.map(r=>['T1',r.round,r.workers,r.wall_s,24,!(r.round===0&&r.workers===1),r.round===0&&r.workers===1?'Suspension : remplacée pour le débit':'Mesure originale conservée',t1.id]);
timingRows.push(['T1-R1',0,1,t1r.data.corrected_summaries.find(r=>r.workers===1).wall_s[0],24,true,'Remplacement prospectif, mêmes résultats',t1r.id]);
table('T1',5,['Étape','Tour','Workers','Durée monotone s','Trajectoires','Retenue débit corrigé','Statut de mesure','Source'],timingRows,{formats:{1:'0',2:'0',3:'0.000',4:'0'},widths:{0:18,3:25,5:28,6:55}});
table('T1',19,['Workers','Durée moyenne corrigée s','Secondes par trajectoire','Accélération vs série','Source'],t1r.data.corrected_summaries.map(r=>[r.workers,null,null,null,t1r.id]),{formats:{0:'0',1:'0.000',2:'0.000000',3:'0.000'},widths:{0:18,1:33,2:33,3:30}});
for(let i=0;i<4;i++){const rr=20+i,r=t1r.data.corrected_summaries[i];formula('T1',`B${rr}`,`=AVERAGEIFS(D6:D14,C6:C14,A${rr},F6:F14,1)`,r.mean_wall_s);formula('T1',`C${rr}`,`=B${rr}/24`,r.mean_s_per_trajectory);formula('T1',`D${rr}`,`=$B$20/B${rr}`,r.speedup_vs_healthy_serial);}

title('F1','F1 : une proposition par parent fixé','6 blocs, deux parents fixés, 4 conditions. Regret de déploiement avec repli seed commun.');
const conditions=['legacy_code','anytime_code','anytime_sparse','anytime_rich'];
const frows=[...f1.data.rows].sort((a,b)=>a.block-b.block||conditions.indexOf(a.condition)-conditions.indexOf(b.condition));
table('F1',5,['Bloc','Condition','Parent','AUC validation','Regret final','Cible atteinte','Temps cible censuré','Éligible train','Trajectoires valides','Trajectoires repli','Statut source','Fin','AUC parent inchangé','AUC seed inchangé','SHA-256 source','Source'],frows.map(r=>[r.block,r.condition,r.parent_kind,r.validation_auc,r.final_regret,r.target_attainment,r.capped_target_evaluations,r.train_valid,r.candidate_valid_trajectories,r.fallback_trajectories,r.source_status,r.finish_reason,r.parent_auc,r.seed_auc,r.source_sha256,f1.id]),{formats:{0:'0',5:'0.00%',8:'0',9:'0'},widths:{1:27,2:22,6:25,8:25,9:25,10:24,12:26,13:26,14:72}});
table('F1',34,['Condition','AUC moyenne','AUC médiane','Regret final moyen','Cible atteinte','Temps cible censuré','Éligibles train','Replis validation','Source'],conditions.map(c=>[c,null,null,null,null,null,null,null,f1.id]),{formats:{4:'0.00%',6:'0',7:'0'},widths:{0:27,3:27,5:25}});
for(let i=0;i<4;i++){const rr=35+i,c=conditions[i],saved=f1.data.conditions[c],locs=frows.flatMap((r,j)=>r.condition===c?[j+6]:[]);formula('F1',`B${rr}`,`=AVERAGEIF(B6:B29,A${rr},D6:D29)`,saved.mean_auc);formula('F1',`C${rr}`,`=MEDIAN(${locs.map(r=>`D${r}`).join(',')})`,saved.median_auc);formula('F1',`D${rr}`,`=AVERAGEIF(B6:B29,A${rr},E6:E29)`,saved.mean_final_regret);formula('F1',`E${rr}`,`=AVERAGEIF(B6:B29,A${rr},F6:F29)`,saved.mean_target_attainment);formula('F1',`F${rr}`,`=AVERAGEIF(B6:B29,A${rr},G6:G29)`,saved.mean_capped_target_evaluations);formula('F1',`G${rr}`,`=COUNTIFS(B6:B29,A${rr},H6:H29,1)`,saved.training_eligible);formula('F1',`H${rr}`,`=SUMIF(B6:B29,A${rr},J6:J29)`,saved.fallback_trajectories);}

title('F1_contrastes','F1 : contrastes appariés enregistrés','Bootstrap importé : 10 000 tirages, RNG 1515, six blocs. Quatre contrastes exploratoires.','IC fragiles et conditionnels aux deux parents et au panel fixé. Delta négatif : première condition meilleure.');
const cs=Object.entries(f1.data.contrasts);
table('F1_contrastes',5,['Contraste',...Array.from({length:6},(_,i)=>String(16301+i)),'Moyenne ΔAUC','Médiane ΔAUC','IC 95 % bas','IC 95 % haut','Lecture enregistrée','Source'],cs.map(([name,r])=>[name,...r.deltas,null,null,...r.paired_bootstrap_95,r.interpretation,f1.id]),{widths:{0:44,7:24,8:24,11:24}});
for(let i=0;i<4;i++){const rr=6+i,[contrast,v]=cs[i],[first,second]=contrast.split('-');for(let j=0;j<6;j++){const block=16301+j,a=frows.findIndex(r=>r.block===block&&r.condition===first)+6,b=frows.findIndex(r=>r.block===block&&r.condition===second)+6;formula('F1_contrastes',`${col(j+1)}${rr}`,`='F1'!D${a}-'F1'!D${b}`,v.deltas[j]);}formula('F1_contrastes',`H${rr}`,`=AVERAGE(B${rr}:G${rr})`,v.mean);formula('F1_contrastes',`I${rr}`,`=MEDIAN(B${rr}:G${rr})`,v.median);}
sheets.F1_contrastes.getRange('B5:G5').setNumberFormat('0');

title('F1_usage','F1 : usage et invalidité de génération','Tous les 24 slots et les 26 tentatives. Raisonnement conservé, jamais ajouté au total.','Coûts des réponses connues. Deux tentatives transport ont une facturation distante potentielle inconnue.');
table('F1_usage',5,['Bloc','Condition','Statut source','Fin','Tokens prompt','Tokens complétion','Tokens raisonnement','Tokens totaux','Coût réponse USD','Coût reçu USD','Tentatives','Complétion distante inconnue','Durée réponse s','Fournisseur','Objectifs effectifs','Allocations inutilisées','Sous-processus','Anomalies compteurs','Source'],frows.map(r=>[r.block,r.condition,r.source_status,r.finish_reason,r.usage.prompt_tokens,r.usage.completion_tokens,r.usage.reasoning_tokens,r.usage.total_tokens,r.usage.cost_usd,r.receipt?.total_cost??null,r.transport_attempts,r.possible_remote_completion_attempts,r.wall_s,r.receipt?.provider_name??null,r.actual_objective_calls,r.unused_objective_allocations,r.subprocess_executions,r.usage_issues.join(' ; '),f1.id]),{formats:{0:'0',4:'#,##0',5:'#,##0',6:'#,##0',7:'#,##0',8:'0.000000000',9:'0.000000000',10:'0',11:'0',12:'0.000',14:'#,##0',15:'#,##0',16:'#,##0'},widths:{1:27,2:24,6:26,8:24,9:24,11:33,13:24,14:26,15:29,17:46}});
table('F1_usage',34,['Condition','Réponses','Tentatives','Tokens prompt','Tokens complétion','Tokens raisonnement','Tokens totaux','Coût réponse USD','Source'],conditions.map(c=>[c,null,null,null,null,null,null,null,f1.id]),{formats:{1:'0',2:'0',3:'#,##0',4:'#,##0',5:'#,##0',6:'#,##0',7:'0.000000000'},widths:{0:27,4:26,5:28,7:25}});
for(let i=0;i<4;i++){const rr=35+i,c=conditions[i],v=f1.data.conditions[c];formula('F1_usage',`B${rr}`,`=COUNTIF(B6:B29,A${rr})`,6);formula('F1_usage',`C${rr}`,`=SUMIF(B6:B29,A${rr},K6:K29)`,v.transport_attempts);for(const [dest,src,key] of [['D','E','prompt_tokens'],['E','F','completion_tokens'],['F','G','reasoning_tokens'],['G','H','total_tokens'],['H','I','cost_usd']])formula('F1_usage',`${dest}${rr}`,`=SUMIF(B6:B29,A${rr},${src}6:${src}29)`,v.usage[key].reported_sum);}

title('Sources','Sources et empreintes des données','Chemins relatifs au dépôt. Les hashes portent sur les fichiers effectivement lus.','Les hashes canoniques des trajectoires B2 sont séparés dans B2 ; les fichiers gzip ont aussi leur hash d’octets.');
table('Sources',5,['ID','Fichier source','Section / champs','SHA-256 des octets','Type d’empreinte','Utilisation'],sourceRows,{widths:{0:12,1:90,2:53,3:75,4:32,5:75}});
sheets.Sources.getRange(`B6:C${5+sourceRows.length}`).format.wrapText=true;
sheets.Sources.getRange(`F6:F${5+sourceRows.length}`).format.wrapText=true;
sheets.Sources.getRange(`A6:F${5+sourceRows.length}`).format.rowHeight=43;
sheets.S1_banque.getRange('B6:B16').format.wrapText=true;
sheets.S1_banque.getRange('A6:O16').format.rowHeight=43;
sheets.Guide.getRange('C6:D13').format.wrapText=true;
sheets.Guide.getRange('A6:D13').format.rowHeight=43;
sheets.Guide.getRange('B18:B24').format.wrapText=true;
sheets.Guide.getRange('A18:B24').format.rowHeight=52;

// Reconcile formula outputs, not only source JSONs, before export.
for(const check of sheetChecks){const actual=sheets[check.sheet].getRange(check.address).values[0][0];close(`${check.sheet}!${check.address}`,actual,check.expected);}
const errors=await wb.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!',options:{useRegex:true,maxResults:50},summary:'Recherche d’erreurs de formule'});
await fs.mkdir(PREVIEWS,{recursive:true});
await fs.writeFile(path.join(HERE,'formula_error_scan.ndjson'),errors.ndjson);
const previewFiles=[];
for(const name of names){
  const primary=sections.find(s=>s.sheet===name);
  const width=Math.min(primary.columns,10);
  const range=`A1:${col(width-1)}${Math.min(primary.end,11)}`;
  const preview=await wb.render({sheetName:name,range,scale:1.35,format:'png'});
  const filename=path.join(PREVIEWS,`${name}.png`);
  await fs.writeFile(filename,new Uint8Array(await preview.arrayBuffer()));previewFiles.push(filename);
  const tableCheck=await wb.inspect({kind:'table',range:`${name}!A${primary.start}:D${Math.min(primary.end,primary.start+2)}`,include:'values,formulas',tableMaxRows:3,tableMaxCols:4,maxChars:1000});
  await fs.writeFile(path.join(HERE,`inspect_${name}.ndjson`),tableCheck.ndjson);
}
// Render the result blocks below long input tables and rightmost wide columns too.
for(const [name,range,label] of [['Guide','A17:B24','Guide_conventions'],['B2','A105:J109','B2_resume'],['G1','A32:H36','G1_resume'],['T1','A18:E24','T1_resume'],['F1','A33:I39','F1_resume'],['F1_usage','A33:I39','F1_usage_resume'],['F1_contrastes','H5:M9','F1_contrastes_IC'],['F1_usage','K5:S10','F1_usage_droite'],['B2','K5:R10','B2_hashes']]){
  const preview=await wb.render({sheetName:name,range,scale:1.35,format:'png'}),filename=path.join(PREVIEWS,`${label}.png`);await fs.writeFile(filename,new Uint8Array(await preview.arrayBuffer()));previewFiles.push(filename);
}
for(const [filename,digest] of inputHashes) assert.equal(sha(await fs.readFile(filename)),digest,'Preserved input changed during export');
await fs.mkdir(path.dirname(OUTPUT),{recursive:true});
await (await SpreadsheetFile.exportXlsx(wb)).save(OUTPUT);
const checks={output:OUTPUT,output_sha256:sha(await fs.readFile(OUTPUT)),builder_sha256:sha(await fs.readFile(fileURLToPath(import.meta.url))),sheets:names,sources:sourceRows.length,sections,numeric_checks:numericChecks,formula_checks:sheetChecks.length,previews:previewFiles,input_byte_hashes:Object.fromEntries(inputHashes),source_font:font,font_availability_check:'fc-list: Liberation Sans available locally',boolean_encoding:'1=yes, 0=no; never a regret imputation',new_model_calls:0,new_objective_calls:0,p1_data_read:false,raw_inputs_unchanged:true};
await fs.writeFile(path.join(HERE,'workbook_checks.json'),JSON.stringify(checks,null,2)+'\n');
console.log(JSON.stringify({output:OUTPUT,sheets:names.length,sources:sourceRows.length,formula_checks:sheetChecks.length,numeric_checks:numericChecks.length,previews:previewFiles.length,sha256:checks.output_sha256}));
