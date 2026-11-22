// Records the real UI and terminal clips used in the demo video (Playwright, 1920x1080).
const {chromium}=require('playwright');const fs=require('fs');const path=require('path');
const [UI_ROOM2,UI_V201,TERM1,TERM2,OUT]=process.argv.slice(2);
async function clip(name,url,ms,setup){const b=await chromium.launch();const ctx=await b.newContext({viewport:{width:1920,height:1080},recordVideo:{dir:OUT+'/'+name,size:{width:1920,height:1080}}});
 const p=await ctx.newPage();if(setup&&setup.init)await p.addInitScript(setup.init);await p.goto(url);if(setup&&setup.after)await p.evaluate(setup.after);await p.waitForTimeout(ms);await ctx.close();await b.close();
 const f=fs.readdirSync(OUT+'/'+name).find(x=>x.endsWith('.webm'));fs.renameSync(OUT+'/'+name+'/'+f,OUT+'/'+name+'.webm');console.log(name)}
const lines=f=>JSON.stringify(fs.readFileSync(f,'utf8').trimEnd().split('\n'));
(async()=>{
 const zoom='document.body.style.zoom=0.9;';
 await clip('room2_play','file://'+UI_ROOM2,22000,{after:zoom});
 await clip('room2_blind','file://'+UI_ROOM2,12000,{after:zoom+"document.getElementById('speed').click();document.getElementById('speed').click();const s=document.getElementById('scrub');s.value=948;s.dispatchEvent(new Event('input'));"});
 await clip('v201_play','file://'+UI_V201,12000,{after:zoom+"const s=document.getElementById('scrub');s.value=300;s.dispatchEvent(new Event('input'));"});
 await clip('term1','file://'+TERM1,16500,{init:'window.LINES='+lines(path.dirname(TERM1)+'/term1.txt')+';window.TYPE=26;window.OUT=260;'});
 await clip('term2','file://'+TERM2,8500,{init:'window.LINES='+lines(path.dirname(TERM2)+'/term2.txt')+';window.TYPE=18;window.OUT=150;'});
})();
