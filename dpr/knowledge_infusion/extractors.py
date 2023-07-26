import dataclasses
import logging
import os
import re
import sqlite3
from abc import ABC, abstractmethod
from hashlib import sha256
from os.path import join as opj
from typing import Dict, List, Tuple, Optional
import numpy as np

import rdflib
import requests
import requests.compat
from SPARQLWrapper import SPARQLWrapper, JSON
from pp_api import PoolParty

logger = logging.getLogger()
null_cpt = "THIS_IS_NOT_A_CONCEPT"
DBPEDIA_SPARQL_ENDPOINT = 'https://dbpedia.org/sparql/'
SPOTLIGHT_LOCAL_ENDPOINT = 'localhost:2222/rest/'
SPOTLIGHT_REMOTE_ENDPOINT = 'https://api.dbpedia-spotlight.org/en/'

class AbstractEntityExtractor(ABC):
    """
    Objects of inheriting classes can be used to find entities in texts.
    """

    @abstractmethod
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def extract_no_overlap(self,
                           text: str,
                           **params) -> Dict[str, List[Tuple[int, int]]]:
        """
        Finds entities in a text, without allowing multiple matches for a
        given offset.
        :param text: The text to extract
        :param params: parameters to be passed to the extraction process
        :return: A dictionary whose keys identify concepts, and whose values
        are lists of start,end offsets where each entity is found

        """
        pass

@dataclasses.dataclass
class ConceptMatch:
    cpt_id: str
    match_label: str
    match_start: int
    match_end: int = None
    cpt_prefLabel: str = None
    cpt_broaderLabel: str = None

    def __post_init__(self):
        if self.match_end is None:
            self.match_end = self.match_start + len(self.match_label)


class PoolPartyExtractor(AbstractEntityExtractor):
    def __init__(self, pphost: str, pppid: str, ppuser: str, pppwd: str,
                 language: str = "en",
                 cpt_id_type: str = "prefLabel",
                 sql_db_tablename: str = "extracted_cpts",
                 use_thes_file: Optional[str] = None,
                 use_uris_as_keys : bool = False,
                 filter_out_preflabel_matches = False,
                 sql_db_path: Optional[str] = None, **kwargs):
        self.cpt_id_type = cpt_id_type
        self.use_uris = use_uris_as_keys
        self.filter_out_preflabel_matches = filter_out_preflabel_matches
        if sql_db_path is None:
            sql_db_path = opj("./tmp",
                              "PP_" + pppid + "_" + ".sqlite3")
            os.makedirs("./tmp", exist_ok=True)
        if not sql_db_path.endswith(".sqlite3"):
            sql_db_path = opj(sql_db_path,"PP_" + pppid + "_" + ".sqlite3")
        self.pp = PoolParty(server=pphost,
                            auth_data=(ppuser, pppwd))
        self.conn = sqlite3.connect(sql_db_path)
        self.thesgraph = None
        if use_thes_file is not None:
            self.thesgraph = rdflib.Graph()
            self.thesgraph.parse(use_thes_file, format="n3")
            logger.info(f"Loaded thesaurus graph from {use_thes_file}"
                        f"with {len(self.thesgraph)} triples")

        self.tablename = sql_db_tablename
        self.default_language = language
        self.default_ppid = pppid
        self._create_tables()
        logger.debug("Finished initializing extractor")

        super().__init__(**kwargs)

    def _create_match_list(self,
                           concept_matches: List[ConceptMatch]
                           ):
        """

        :param concept_matches:
        :return: a dictionary whose keys are concepts and whose values are
        lists of pairs of start,end offsets. E.g.
        {
         "cpt1":[(1,5),(21,25)],
         "cpt2":[(7,10),(37,49)]
        }
        """
        if len(concept_matches)==0:
            return dict()
        maxend = max([cm.match_end for cm in concept_matches])
        hitmap = np.zeros(maxend+100)
        result = dict()
        for cm in concept_matches:
            if self.use_uris:
                cpt = cm.cpt_id
            elif self.cpt_id_type == "prefLabel":
                cpt = cm.cpt_prefLabel
            elif self.cpt_id_type == "broaderLabel":
                cpt = cm.cpt_broaderLabel
            else:
                raise ValueError(f"{self.cpt_id_type} is not a valid cpt_id_type")
            beg = cm.match_start
            end = cm.match_end
            hitmap[beg:end+1] += 1
            if hitmap[beg:end+1].max() > 1:
                continue
            thiscpt = result.get(cpt, [])
            thiscpt.append((beg, end))
            result[cpt] = thiscpt
        return result

    def _find_matches_in_db(self, document_id):
        query = f"SELECT cpt_id, cpt_prefLabel, matched_label, begin, end, " \
                f"cpt_broaderLabel " \
                f"FROM {self.tablename} " \
                f"WHERE doc_id='{document_id}'"
        cur = self.conn.cursor()
        res = cur.execute(query)
        matches = [ConceptMatch(cpt_id=r[0],
                                cpt_prefLabel=r[1],
                                match_label=r[2],
                                match_start=r[3],
                                match_end=r[4],
                                cpt_broaderLabel=r[5]) for r in res.fetchall()]
        result = self._create_match_list(matches)
        return result

    def _document_exists_in_db(self, docid):
        q = f""" 
        SELECT doc_id,full_text 
        FROM documents 
        WHERE doc_id='{docid}'
        """
        cur = self.conn.cursor()
        res = cur.execute(q).fetchall()
        return len(res) > 0

    def _store_document_into_db(self, docid, text):
        query = f"INSERT INTO documents" \
                f"(doc_id, full_text)" \
                f"VALUES (?, ?); "
        values = [(docid, text)]
        try:
            cur = self.conn.cursor()
            rs = cur.executemany(query, values)
        finally:
            if cur is not None:
                self.conn.commit()
                cur.close()
                logger.debug(f"Stored document {docid} into database")

    def _store_extraction_into_db(self,
                                  docid: str,
                                  concept_matches: List[ConceptMatch]):
        query = f"INSERT INTO {self.tablename}" \
                f"(doc_id, cpt_id, cpt_prefLabel, cpt_broaderLabel,  " \
                f"matched_label, begin, " \
                f"end)" \
                f"VALUES (?, ?, ?, ?, ?, ?, ?); "
        values = [(docid,
                   c.cpt_id, c.cpt_prefLabel, c.cpt_broaderLabel,
                   c.match_label, c.match_start, c.match_end)
                  for c in concept_matches]
        try:
            cur = self.conn.cursor()
            # print(values)
            rs = cur.executemany(query, values)
        finally:
            if cur is not None:
                self.conn.commit()
                cur.close()

    def _create_doc_id(self, text):
        starttext: str = text[:20].replace(" ", "_").replace("\n", "_")
        starttext = "".join([ch for ch in starttext
                             if ch.isalnum()
                             or ch == "_"])
        docid = starttext + str(sha256(text.encode('utf-8')).hexdigest())
        return docid

    def _results2broaderLabel(self, cpt, occ, lang):

        cptlocalname = cpt["uri"].split("/")[-1]
        if "#" in cptlocalname:
            cptlocalname = cptlocalname.split("#")[-1]
        if len(cpt.get("broaderConcepts", [])) > 0:
            broader_uri = cpt.get("broaderConcepts", [])[0]
            broaderlabel = self._get_label_from_db(broader_uri)
            lab = broaderlabel
        else:
            cpt_scheme = cpt.get("conceptSchemes", [])[0]
            lab = cpt_scheme.get("title", cptlocalname)
        return lab

    def _results2prefLabel(self, cpt, occ, lang):
        lab = cpt["prefLabels"].get(lang, occ['matchedText'])
        self._store_label_into_db(cpt['uri'], lab)

        return lab

    def _get_label_from_db(self, cpt, pid=None, lang=None):
        if pid is None:
            pid = self.default_ppid
        if lang is None:
            lang = self.default_language
        q = f""" 
                SELECT prefLabel
                FROM labels
                WHERE cpt_id='{cpt}'
                """
        cur = self.conn.cursor()
        res = cur.execute(q).fetchall()
        if len(res) > 0:
            label = [r[0] for r in res][0]
        else:
            cptinfo = []
            if self.thesgraph is None:
                cptinfo = self.pp.get_cpts_info(uris=[cpt],
                                                pid=pid,
                                                lang=lang)
                if len(cptinfo) == 0:
                    self.pp.get_cpts_info(uris=[cpt],
                                          pid=pid)
            else:
                labs = [x for x in self.thesgraph.triples((rdflib.URIRef(cpt),
                                               rdflib.namespace.SKOS[
                                                   "prefLabel"],
                                               None))]
                for _, _, lab in labs:
                    lab: rdflib.Literal
                    if lab.language == lang or len(labs) == 0:
                        cptinfo.append({"prefLabel": lab.value})

            if len(cptinfo) > 0:
                label = cptinfo[0]["prefLabel"]
            else:
                label = cpt.split("/")[-1]
                if "#" in label:
                    label = label.split("#")[-1]
            self._store_label_into_db(cpt, label)
        return label

    def _store_label_into_db(self, cpt, label):
        q = "INSERT INTO labels (cpt_id, prefLabel) VALUES (?,?);  "
        values = [(cpt, label)]
        try:
            cur = self.conn.cursor()
            # print(values)
            rs = cur.executemany(q, values)
        finally:
            if cur is not None:
                self.conn.commit()
                cur.close()

    def extract_no_overlap(self,
                           text: str,
                           ppid: Optional[str] = None,
                           lang: Optional[str] = None,
                           force_extraction=False,
                           **params) -> Dict[str, List[Tuple[int, int]]]:
        """

        :param text: The text to extract
        :param ppid: Project ID, if None, the extractors default is used
        :param lang: Language of the text, if None, the extractors default
        :param force_extraction:  If True, extraction will be executed even
        if the result is in the Cache
        :param params:
        :return: a dictionary whose keys are concepts and whose values are
        lists of pairs of start,end offsets. E.g.
        {
         "cpt1":[(1,5),(21,25)],
         "cpt2":[(7,10),(37,49)]
        }
        """
        docid = self._create_doc_id(text)
        from_table = self._find_matches_in_db(docid)
        if (not force_extraction and
                (len(from_table) > 0
                 or self._document_exists_in_db(docid))):
            logger.debug(f"Cache hit for document {docid}"
                         f" with len {len(from_table)} "
                         f" doc_exists {self._document_exists_in_db(docid)}")
            return from_table

        logger.debug(f"Cache miss for document {docid}")
        if ppid is None:
            ppid = self.default_ppid
        if lang is None:
            lang = self.default_language


        logger.debug(f"Sending {docid} for extraction")
        ex_res = self.pp.extract(text=text, pid=ppid, lang=lang,
                                 force_json=True)
        ex_data = ex_res.json()
        # print(ex_data.keys(),"\n---\n")
        matches = []
        if 'concepts' in ex_data:
            ex_cpts = ex_data['concepts']
            logger.debug(f"got {len(ex_cpts)} concepts back")
            matches += [ConceptMatch(cpt_id=cpt["uri"],
                                     cpt_prefLabel=self._results2prefLabel(
                                         cpt, occ, lang),
                                     cpt_broaderLabel=self._results2broaderLabel(
                                         cpt, occ, lang),
                                     match_end=pos['endIndex'],
                                     match_start=pos['beginningIndex'],
                                     match_label=occ['matchedText'])

                        for cpt in ex_cpts
                        for ml in cpt['matchingLabels']
                        for occ in ml['matchedTexts']
                        for pos in occ['positions']
                        if (not self.filter_out_preflabel_matches or
                            self._results2prefLabel(cpt, occ, lang).lower() != occ['matchedText'].lower())
                        ]
        # print("Matches...............")
        # print(json.dumps(matches,indent=2))
        # print(".............|")
        self._store_extraction_into_db(docid=docid,
                                       concept_matches=matches)
        self._store_document_into_db(docid=docid,
                                     text=text)

        result = self._create_match_list(concept_matches=matches)
        return result

    def _create_tables(self):
        cur = None
        try:
            # ----- Matches
            cur = self.conn.cursor()
            sql = f"""
            CREATE TABLE IF NOT EXISTS {self.tablename} (
                "doc_id"	TEXT NOT NULL,
                "cpt_id"	TEXT NOT NULL,
                "cpt_prefLabel"  TEXT,
                "cpt_broaderLabel"  TEXT,  
                "matched_label"	TEXT,
                "begin"	INTEGER,
                "end"	INTEGER,                
                UNIQUE(doc_id, begin, end) ON CONFLICT IGNORE
            );            
            """
            rs = cur.execute(sql)
            sql = f"CREATE INDEX  IF NOT EXISTS doc_index ON" \
                  f" {self.tablename}(doc_id);"
            rs = cur.execute(sql)

            # ---- Documents
            sql = f"""
            CREATE TABLE IF NOT EXISTS documents (
                "doc_id"    TEXT NOT NULL,
                "full_text" BLOB 
            );
            """
            rs = cur.execute(sql)

            # ---- Cpt Labels
            sql = f"""
                        CREATE TABLE IF NOT EXISTS labels (
                            "cpt_id"     TEXT NOT NULL,
                            "prefLabel" TEXT ,
                             UNIQUE(cpt_id, prefLabel) ON CONFLICT REPLACE
                        );
                        """
            rs = cur.execute(sql)

        finally:
            if cur is not None:
                self.conn.commit()
                cur.close()
        return rs


class SpotlightExtractor(AbstractEntityExtractor):
    def __init__(self, language: str = "en",
                 cpt_id_type: str = "prefLabel",
                 sql_db_tablename: str = "extracted_cpts",
                 sql_db_path: Optional[str] = None, **kwargs):
        self.cpt_id_type = cpt_id_type
        if sql_db_path is None:
            sql_db_path = opj("/tmp",
                              "DBpedia.sqlite3")
        else:
            if not sql_db_path.endswith(".sqlite3"):
                sql_db_path = opj(sql_db_path,"DBpedia.sqlite3")
        self.conn = sqlite3.connect(sql_db_path)
        self.tablename = sql_db_tablename
        self.default_language = language
        self.create_tables()
        logger.debug("Finished initializing extractor")

    def create_tables(self):
        cur = None
        try:
            # ----- Matches
            cur = self.conn.cursor()
            sql = f"""
            CREATE TABLE IF NOT EXISTS {self.tablename} (
                "doc_id"	TEXT NOT NULL,
                "cpt_id"	TEXT NOT NULL,
                "cpt_prefLabel"  TEXT,
                "cpt_broaderLabel"  TEXT,  
                "matched_label"	TEXT,
                "begin"	INTEGER,
                "end"	INTEGER,                
                UNIQUE(doc_id, begin, end) ON CONFLICT IGNORE
            );            
            """
            rs = cur.execute(sql)
            sql = f"CREATE INDEX IF NOT EXISTS doc_index ON" \
                  f" {self.tablename}(doc_id);"
            rs = cur.execute(sql)
            # ---- Documents
            sql = f"""
            CREATE TABLE IF NOT EXISTS documents (
                "doc_id"    TEXT NOT NULL,
                "full_text" BLOB 
            );
            """
            rs = cur.execute(sql)
            # ---- Cpt Labels
            sql = f"""
                        CREATE TABLE IF NOT EXISTS labels (
                            "cpt_id"     TEXT NOT NULL,
                            "prefLabel" TEXT ,
                             UNIQUE(cpt_id, prefLabel) ON CONFLICT REPLACE
                        );
                        """
            rs = cur.execute(sql)
        finally:
            if cur is not None:
                self.conn.commit()
                cur.close()
        return rs

    def _create_match_list(self, concept_matches: List[ConceptMatch]):
        """
        :param concept_matches:
        :return: a dictionary whose keys are concepts and whose values are
        lists of pairs of start,end offsets. E.g.
        {
         "cpt1":[(1,5),(21,25)],
         "cpt2":[(7,10),(37,49)]
        }
        """
        result = dict()
        for cm in concept_matches:
            cpt = cm.cpt_prefLabel
            if self.cpt_id_type == "broaderLabel":
                cpt = cm.cpt_broaderLabel
            beg = cm.match_start
            end = cm.match_end
            thiscpt = result.get(cpt, [])
            thiscpt.append((beg, end))
            result[cpt] = thiscpt
        return result

    def _find_matches_in_db(self, document_id):
        query = f"SELECT cpt_id, cpt_prefLabel, matched_label, begin, end, " \
                f"cpt_broaderLabel " \
                f"FROM {self.tablename} " \
                f"WHERE doc_id='{document_id}'"
        cur = self.conn.cursor()
        res = cur.execute(query)
        matches = [ConceptMatch(cpt_id=r[0],
                                cpt_prefLabel=r[1],
                                match_label=r[2],
                                match_start=r[3],
                                match_end=r[4],
                                cpt_broaderLabel=r[5]) for r in res.fetchall()]
        result = self._create_match_list(matches)
        return result

    def _document_exists_in_db(self, docid):
        q = f""" 
        SELECT doc_id,full_text 
        FROM documents 
        WHERE doc_id='{docid}'
        """
        cur = self.conn.cursor()
        res = cur.execute(q).fetchall()
        return len(res) > 0

    def _store_document_into_db(self, docid, text):
        query = f"INSERT INTO documents" \
                f"(doc_id, full_text)" \
                f"VALUES (?, ?); "
        values = [(docid, text)]
        cur = None
        try:
            cur = self.conn.cursor()
            rs = cur.executemany(query, values)
        finally:
            if cur is not None:
                self.conn.commit()
                cur.close()
                logger.debug(f"Stored document {docid} into database")

    def _store_extraction_into_db(self,
                                  docid: str,
                                  concept_matches: List[ConceptMatch]):
        query = f"INSERT INTO {self.tablename}" \
                f"(doc_id, cpt_id, cpt_prefLabel, cpt_broaderLabel,  " \
                f"matched_label, begin, " \
                f"end)" \
                f"VALUES (?, ?, ?, ?, ?, ?, ?); "
        values = [(docid,
                   c.cpt_id, c.cpt_prefLabel, c.cpt_broaderLabel,
                   c.match_label, c.match_start, c.match_end)
                  for c in concept_matches]
        cur = None
        try:
            cur = self.conn.cursor()
            rs = cur.executemany(query, values)
        finally:
            if cur is not None:
                self.conn.commit()
                cur.close()

    @staticmethod
    def _create_doc_id(text):
        starttext: str = text[:20].replace(" ", "_").replace("\n", "_")
        starttext = "".join([ch for ch in starttext if (ch.isalnum() or ch == "_")])
        docid = starttext + str(sha256(text.encode('utf-8')).hexdigest())
        return docid

    @staticmethod
    def query_dbpedia_canonical_label(match_uri):
        name = match_uri.split("/")[-1].replace("_"," ")
        sparql = SPARQLWrapper(DBPEDIA_SPARQL_ENDPOINT)
        sparql.setQuery(f"""
                PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                SELECT DISTINCT ?name ?cpt
                {{
                    <{match_uri}> dbp:name|rdfs:label ?name 
                    FILTER langMatches( lang(?name), "EN" )
                }}""")
        sparql.setReturnFormat(JSON)
        results = sparql.query().convert()
        if ("results" in results.keys()
                and "bindings" in results["results"].keys()
                and len(results["results"]["bindings"])>0):
            name = results["results"]["bindings"][0]["name"]["value"]
        return name

    # def _results2prefLabel(self, cpt, occ, lang):
    #     lab = cpt["prefLabels"].get(lang, occ['matchedText'])
    #     self._store_label_into_db(cpt['uri'], lab)
    #     return lab
    #
    # def _results2broaderLabel(self, cpt, occ, lang):
    #     cptlocalname = cpt["uri"].split("/")[-1]
    #     if "#" in cptlocalname:
    #         cptlocalname = cptlocalname.split("#")[-1]
    #     if len(cpt.get("broaderConcepts", [])) > 0:
    #         broader_uri = cpt.get("broaderConcepts", [])[0]
    #         broaderlabel = self._get_label_from_db(broader_uri)
    #         lab = broaderlabel
    #     else:
    #         cpt_scheme = cpt.get("conceptSchemes", [])[0]
    #         lab = cpt_scheme.get("title",cptlocalname)
    #     return lab
    def _get_label_from_db(self, cpt_id):
        cpt_id = cpt_id.replace("'","_")
        q = f"""
                SELECT prefLabel
                FROM labels
                WHERE cpt_id='{cpt_id}'
                """
        cur = self.conn.cursor()
        res = cur.execute(q).fetchall()
        if len(res) > 0:
            label = [r[0] for r in res][0]
        else:
            label = None
        return label

    def _store_label_into_db(self, cpt, label):
        cpt = cpt.replace("'","_")
        q = "INSERT INTO labels (cpt_id, prefLabel) VALUES (?,?);  "
        values = [(cpt, label)]
        cur = None
        try:
            cur = self.conn.cursor()
            rs = cur.executemany(q, values)
        finally:
            if cur is not None:
                self.conn.commit()
                cur.close()

    def extract_no_overlap(self,
                           text: str,
                           lang: Optional[str] = 'en',
                           force_extraction=False,
                           **params) -> Dict[str, List[Tuple[int, int]]]:
        """

        :param text: The text to extract
        :param lang: Language of the text. Currently only works with English.
        :param force_extraction:  If True, extraction will be executed even
                if the result is in the Cache
        :param params:
        :return: a dictionary whose keys are concepts and whose values are
                lists of pairs of start,end offsets. E.g.
                {
                 "cpt1":[(1,5),(21,25)],
                 "cpt2":[(7,10),(37,49)]
                }
        """
        docid = self._create_doc_id(text)
        from_table = self._find_matches_in_db(docid)
        if not force_extraction and (len(from_table) > 0 or self._document_exists_in_db(docid)):
            logger.debug(f"Cache hit for document {docid}"
                         f" with len {len(from_table)} "
                         f" doc_exists {self._document_exists_in_db(docid)}")
            return from_table
        logger.debug(f"Cache miss for document {docid}")
        self._store_document_into_db(docid=docid, text=text)
        logger.debug(f"Send {docid} for extraction")
        matches = self.spot(text=text)
        for cm in matches:
            cached_label = self._get_label_from_db(cm.cpt_id)
            logger.debug(f"Cached label for {cm.cpt_id} = {cached_label}.")
            if cached_label is None:
                cached_label = self.query_dbpedia_canonical_label(cm.cpt_id)
                assert cached_label is not None, cached_label
                self._store_label_into_db(cpt=cm.cpt_id, label=cached_label)
                logger.debug(f"Queried DBpedia, found {cached_label}, stored")
            cm.cpt_prefLabel = cached_label
        logger.debug(f"Ingest {docid} into store")
        self._store_extraction_into_db(docid=docid, concept_matches=matches)
        logger.debug(f"prepare output")
        result = self._create_match_list(concept_matches=matches)
        return result

    @staticmethod
    def spot(text: str, endpoint=SPOTLIGHT_REMOTE_ENDPOINT) -> List[ConceptMatch]:
        headers = {
            'Accept': 'application/json',
            'Content-Type': 'application/x-www-form-urlencoded',
        }
        data = {
            'text': text,
            'confidence': '0.35',
        }
        response = requests.post(requests.compat.urljoin(endpoint, 'annotate'),
                                 headers=headers, data=data)
        response.raise_for_status()
        json_response = response.json()
        # prepare output
        out = []
        if 'Resources' not in json_response.keys():
            return []
        for ann in json_response['Resources']:
            cpt_id = ann['@URI']
            match_label = ann['@surfaceForm']
            match_start = int(ann['@offset'])
            # match_end is computed as post_init
            cm = ConceptMatch(cpt_id=cpt_id, match_label=match_label,
                              match_start=match_start)
            out.append(cm)
        return out


class DummyExtractor(AbstractEntityExtractor):
    def extract_no_overlap(self, text: str, **params):
        result = dict()
        for m in re.finditer(r'\S+', text):
            start = m.start()
            end = m.end()
            word = text[start:end]
            startchr = word[0].lower()
            if startchr in self.firstchardict.keys():
                concept = self.firstchardict[startchr]
                matches_this_concept = result.get(concept, [])
                matches_this_concept.append((start, end))
                result[concept] = matches_this_concept
        return result

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.firstchardict = {"a": "aardvark",
                              "i": "impala",
                              "r": "racoon",
                              "o": "octopus",
                              "k": "koala",
                              "l": "lynx",
                              "p": "possum",
                              "g": "gnu",
                              "c": "cat",
                              "d": "dog",
                              "s": "starfish"}


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    dbe = SpotlightExtractor()
    dbe.extract_no_overlap(text="At a very basic level, primary flight displays take data from sensors, "
                                "do calculations, and display the results on a screen. Of course, because "
                                "they are digital computers, the input must be digital, at least inside the "
                                "processor. However, the values being measured, such as air pressure, airspeed, "
                                "etc., are fundamentally analog, so there must be analog sensors involved. At "
                                "what point is the analog data from the sensors converted to digital data for "
                                "the computer?")
