// Page de l'objectif 1 : segmentation, résumé, tableau des zones cliquables, zone agrandie.
(() => {
  const seuil = document.getElementById("seuil");
  const seuilVal = document.getElementById("seuil-val");
  const resume = document.getElementById("resume");
  const imgResultat = document.getElementById("img-resultat");
  const imgCarte = document.getElementById("img-carte");
  const imgZoom = document.getElementById("img-zoom");
  const calque = document.getElementById("calque-zones");
  const corps = document.querySelector("#tableau-zones tbody");
  const infos = document.getElementById("infos-zone");
  const MESSAGE = "Déposez une image ou choisissez un exemple, puis cliquez sur « Analyser ».";
  const CLASSE_NIVEAU = { "élevée": "eleve", "moyenne": "moyen", "faible": "faible" };
  let resultat = null;               // dernière réponse du serveur
  let enCours = false;

  // ---------- onglets ----------
  document.querySelectorAll(".onglet").forEach(o => o.addEventListener("click", () => {
    document.querySelectorAll(".onglet").forEach(x => x.classList.toggle("actif", x === o));
    document.querySelectorAll(".vue").forEach(v => v.hidden = v.id !== o.dataset.cible);
  }));

  // ---------- affichage ----------
  function stat(valeur, libelle, couleur) {
    return `<div class="stat" style="--s:${couleur}"><div class="gros">${valeur}</div><div class="petit">${libelle}</div></div>`;
  }

  function messageResume(texte, alerte = false) {
    resume.className = "resume vide" + (alerte ? " alerte" : "");
    resume.innerHTML = `<span>${echapper(texte)}</span>`;
  }

  function vider() {
    resultat = null;
    messageResume(MESSAGE);
    afficherImage(imgResultat, null);
    afficherImage(imgCarte, null);
    calque.innerHTML = "";
    corps.innerHTML = `<tr class="ligne-vide"><td colspan="5">Aucune analyse pour l'instant.</td></tr>`;
    viderZoom("Sélectionnez une zone pour l'agrandir.");
  }

  function viderZoom(texte) {
    afficherImage(imgZoom, null);
    imgZoom.closest(".cadre-image").querySelector(".vide-image").textContent = texte;
    infos.innerHTML = "";
  }

  function afficher(r) {
    resultat = r;
    const s = r.nb_zones > 1 ? "s" : "";
    if (r.nb_zones === 0) {
      messageResume(`Aucune anomalie au seuil ${r.seuil.toFixed(2)} · analyse en ${r.ms} ms. Essayez de baisser le seuil.`);
    } else {
      resume.className = "resume";
      resume.innerHTML = stat(r.nb_zones, `zone${s} détectée${s}`, "#F95738")
        + stat(r.surface, "de l'image", "#2563EB")
        + stat(r.conf_max, "confiance max.", "#16A34A")
        + stat(`${r.ms} ms`, "temps d'analyse", "#7C3AED");
    }
    afficherImage(imgResultat, r.resultat);
    afficherImage(imgCarte, r.carte);

    calque.innerHTML = r.zones.map((z, i) => {
      const [x, y, w, h] = z.cadre.map(v => (v * 100).toFixed(3) + "%");
      return `<div class="boite" data-i="${i}" title="Zone ${z.numero}" style="left:${x};top:${y};width:${w};height:${h}"></div>`;
    }).join("");
    calque.querySelectorAll(".boite").forEach(b => b.addEventListener("click", () => choisirZone(+b.dataset.i)));

    corps.innerHTML = r.zones.length ? r.zones.map((z, i) => `
      <tr class="zone" data-i="${i}">
        <td><span class="num-zone">${z.numero}</span></td>
        <td>${echapper(z.position)}</td>
        <td>${z.surface}</td>
        <td>${z.confiance}</td>
        <td><span class="niveau ${CLASSE_NIVEAU[z.niveau] || ""}">${z.niveau}</span></td>
      </tr>`).join("")
      : `<tr class="ligne-vide"><td colspan="5">Aucune zone au-dessus du seuil.</td></tr>`;
    corps.querySelectorAll("tr.zone").forEach(tr => tr.addEventListener("click", () => choisirZone(+tr.dataset.i)));

    if (r.zones.length) choisirZone(0, false);
    else viderZoom("Aucune zone à agrandir.");
  }

  function choisirZone(i, surbrillance = true) {
    const z = resultat?.zones[i];
    if (!z) return;
    corps.querySelectorAll("tr.zone").forEach(tr => tr.classList.toggle("choisie", +tr.dataset.i === i));
    calque.querySelectorAll(".boite").forEach(b => b.classList.toggle("choisie", surbrillance && +b.dataset.i === i));
    afficherImage(imgZoom, z.zoom);
    infos.innerHTML = `<dl>
      <dt>Zone</dt><dd>n° ${z.numero}</dd>
      <dt>Position</dt><dd>${echapper(z.position)}</dd>
      <dt>Taille</dt><dd>${z.taille} · ${z.surface} de l'image</dd>
      <dt>Confiance</dt><dd>${z.confiance} en moyenne (${z.niveau}) · ${z.conf_max} au maximum</dd>
    </dl>`;
    if (surbrillance) {
      // revenir sur l'onglet « Anomalies détectées » pour voir la zone en surbrillance
      document.querySelector('.onglet[data-cible="vue-resultat"]').click();
    }
  }

  // ---------- analyse ----------
  async function analyser(seulementSeuil = false) {
    if (enCours || !Entree.source) return;
    enCours = true;
    Entree.chargement(true);
    try {
      const champs = { seuil: seuil.value };
      let rep;
      if (seulementSeuil && resultat) {            // même image : le serveur la garde en mémoire
        rep = await Entree.envoyer({ ...champs, id: resultat.id }, false);
        if (rep.statut === 410) rep = await Entree.envoyer(champs);
      } else {
        rep = await Entree.envoyer(champs);
      }
      if (!rep.ok) messageResume(rep.json.erreur || "Erreur inconnue.", true);
      else afficher(rep.json);
    } catch (e) {
      messageResume("Le serveur ne répond pas. Vérifiez que « python app.py » tourne toujours.", true);
    } finally {
      enCours = false;
      Entree.chargement(false);
    }
  }

  Entree.bouton.addEventListener("click", () => analyser(false));
  Entree.surChangement(() => { resultat = null; analyser(false); });   // analyse dès qu'une image est choisie
  Entree.surEffacement(vider);
  seuil.addEventListener("input", () => seuilVal.textContent = (+seuil.value).toFixed(2));
  seuil.addEventListener("change", () => { if (resultat) analyser(true); });
})();
