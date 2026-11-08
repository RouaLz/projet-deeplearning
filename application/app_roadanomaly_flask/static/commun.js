// Partie commune à toutes les pages : choix de l'image (fichier, glisser-déposer, exemple) et envoi au serveur.
const Entree = (() => {
  const depot = document.getElementById("depot");
  const fichier = document.getElementById("fichier");
  const apercu = document.getElementById("apercu");
  const btnAnalyser = document.getElementById("analyser");
  const btnEffacer = document.getElementById("effacer");
  const exemples = document.querySelectorAll(".exemple");
  let source = null;                 // {fichier: File} ou {exemple: "nom.jpg"}
  const ecouteurs = { change: [], efface: [] };

  function choisir(src, urlApercu) {
    source = src;
    apercu.src = urlApercu;
    apercu.hidden = false;
    depot.classList.add("rempli");
    btnAnalyser.disabled = false;
    exemples.forEach(e => e.classList.toggle("choisi", src.exemple === e.dataset.nom));
    ecouteurs.change.forEach(f => f());
  }

  function choisirFichier(f) {
    if (!f || !f.type.startsWith("image/")) return;
    choisir({ fichier: f }, URL.createObjectURL(f));
  }

  function effacer() {
    source = null;
    apercu.hidden = true;
    apercu.removeAttribute("src");
    fichier.value = "";
    depot.classList.remove("rempli");
    btnAnalyser.disabled = true;
    exemples.forEach(e => e.classList.remove("choisi"));
    ecouteurs.efface.forEach(f => f());
  }

  fichier.addEventListener("change", () => choisirFichier(fichier.files[0]));
  ["dragenter", "dragover"].forEach(t => depot.addEventListener(t, e => { e.preventDefault(); depot.classList.add("survol"); }));
  ["dragleave", "drop"].forEach(t => depot.addEventListener(t, e => { e.preventDefault(); depot.classList.remove("survol"); }));
  depot.addEventListener("drop", e => choisirFichier(e.dataTransfer.files[0]));
  document.addEventListener("paste", e => {           // Ctrl + V d'une image copiée
    const item = [...(e.clipboardData?.items || [])].find(i => i.type.startsWith("image/"));
    if (item) choisirFichier(item.getAsFile());
  });
  exemples.forEach(e => e.addEventListener("click", () => choisir({ exemple: e.dataset.nom }, e.querySelector("img").src)));
  btnEffacer.addEventListener("click", effacer);

  // Envoie l'image (ou d'autres champs) au serveur et renvoie la réponse JSON
  async function envoyer(champs = {}, avecImage = true) {
    const donnees = new FormData();
    if (avecImage && source?.fichier) donnees.append("image", source.fichier);
    if (avecImage && source?.exemple) donnees.append("exemple", source.exemple);
    for (const [k, v] of Object.entries(champs)) donnees.append(k, v);
    const rep = await fetch(`/api/objectif/${window.NUMERO}/analyser`, { method: "POST", body: donnees });
    let json = {};
    try { json = await rep.json(); } catch { json = { erreur: `Réponse inattendue du serveur (${rep.status}).` }; }
    return { statut: rep.status, ok: rep.ok, json };
  }

  function chargement(actif) {
    btnAnalyser.classList.toggle("charge", actif);
    btnAnalyser.disabled = actif || !source;
  }

  return {
    get source() { return source; },
    envoyer, chargement, effacer,
    surChangement: f => ecouteurs.change.push(f),
    surEffacement: f => ecouteurs.efface.push(f),
    bouton: btnAnalyser,
  };
})();

function afficherImage(img, url) {
  const cadre = img.closest(".cadre-image");
  if (url) { img.src = url; img.hidden = false; cadre.classList.add("a-image"); }
  else { img.hidden = true; img.removeAttribute("src"); cadre.classList.remove("a-image"); }
}

function echapper(t) {
  return String(t).replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}
