"""Сценарії інцидентів для симулятора.

Кожен сценарій — клас з трьома методами:
    start(state, params, rng)                  — що стається в момент початку
    apply(state, values, minute, params, rng)  — як змінюються значення щохвилини
    end(state, params, rng)                    — як сервер повертається в норму

state   — стан сервера (ServerState із simulator.py): служби, теки, PID процесів...
values  — значення поточної хвилини (CPU, RAM, HTTP...), які сценарій може змінити
minute  — скільки хвилин минуло від початку інциденту (0, 1, 2, ...)
params  — дані саме цього інциденту:
            duration_min — тривалість, хв (в історії — випадкова,
                           у живому режимі — з --duration або live_duration_min)
            live         — True у живому режимі (кроки 1.6–1.7)
            інші поля — те, що сценарій запам'ятовує для себе

Кожен метод повертає список подій журналу Windows (може бути порожнім).
Ті самі класи працюють і для історії, і для живого режиму, тож інциденти
на демо виглядають так само, як ті, на яких вчиться ML.
"""

from datetime import timedelta


def event(log: str, source: str, event_id: int, level: str, message: str) -> dict:
    """Подія журналу Windows (час і сервер додає симулятор)."""
    return {"log": log, "source": source, "event_id": event_id,
            "level": level, "message": message}


class Scenario:
    """Базовий клас: сценарій, який нічого не робить."""
    name = ""
    duration_min = (10, 30)    # тривалість в історії, хв (від, до)
    live_duration_min = 10     # тривалість у живому режимі за замовчуванням

    def start(self, state, params, rng) -> list[dict]:
        return []

    def apply(self, state, values, minute, params, rng) -> list[dict]:
        return []

    def end(self, state, params, rng) -> list[dict]:
        return []

    @staticmethod
    def target(state, name: str) -> dict:
        """Що саме ламає сценарій на цьому сервері (з profiles.yaml)."""
        return state.profile.get("scenario_targets", {}).get(name, {})


# ---------------------------------------------------------------------------

class CpuSpike(Scenario):
    """CPU 92–98 %: один процес «з'їдає» процесор."""
    name = "cpu_spike"
    duration_min = (10, 40)

    def apply(self, state, values, minute, params, rng):
        values["cpu_pct"] = rng.uniform(92, 98)
        if values.get("app_response_ms") is not None:
            values["app_response_ms"] *= 1.8      # сайт сповільнюється
        if values.get("http"):
            values["http"]["response_ms"] *= 1.8
        t = self.target(state, self.name)
        params.setdefault("process", t.get("process"))     # для «ключа відповідей»
        params.setdefault("app_pool", t.get("app_pool"))
        values["process_overrides"].append({
            "name": t.get("process"), "app_pool": t.get("app_pool"),
            "cpu_pct": values["cpu_pct"] * rng.uniform(0.75, 0.85),
        })
        return []


class MemoryLeak(Scenario):
    """Витік пам'яті: RAM рівномірно росте до ~95 %, потім процес перезапускають."""
    name = "memory_leak"
    duration_min = (120, 360)

    def apply(self, state, values, minute, params, rng):
        if "start_ram" not in params:
            params["start_ram"] = values["ram_pct"]   # RAM у момент початку витоку
            params["warned"] = False
        # RAM рівномірно росте від стартового значення до ~95 % до кінця інциденту
        # (не залежить від добового циклу — витік лише накопичується)
        progress = min(minute / max(params["duration_min"], 1), 1.0)
        leak_level = params["start_ram"] + (95 - params["start_ram"]) * progress
        values["ram_pct"] = min(leak_level + rng.gauss(0, 0.2), 99)
        extra_pct = leak_level - params["start_ram"]

        t = self.target(state, self.name)
        params.setdefault("process", t.get("process"))     # для «ключа відповідей»
        params.setdefault("app_pool", t.get("app_pool"))
        values["process_overrides"].append({
            "name": t.get("process"), "app_pool": t.get("app_pool"),
            "ram_add_mb": extra_pct / 100 * state.profile["ram_total_gb"] * 1024,
        })

        events = []
        if values["ram_pct"] >= 93 and not params["warned"]:
            params["warned"] = True
            pid = state.pid_for_target(t)
            events.append(event(
                "System", "Microsoft-Windows-Resource-Exhaustion-Detector", 2004, "Warning",
                "Windows successfully diagnosed a low virtual memory condition. "
                "The following programs consumed the most virtual memory: "
                f"{t.get('process')} ({pid}) consumed "
                f"{int(state.profile['ram_total_gb'] * 0.6 * 1024 ** 3)} bytes."))
        return events

    def end(self, state, params, rng):
        # «Перезапуск» процесу: новий PID, пам'ять повертається в норму
        state.restart_target(self.target(state, self.name), rng)
        return []


class DiskFill(Scenario):
    """Тека некеровано росте, диск заповнюється до ~93 %, потім «прибирання».

    Тека — з params["folder"] (для ланцюжків) або з scenario_targets профілю.
    """
    name = "disk_fill"
    duration_min = (120, 360)

    def apply(self, state, values, minute, params, rng):
        if "rate_gb_per_min" not in params:
            folder = params.get("folder") or self.target(state, self.name)["folder"]
            drive = state.folder_drive(folder)
            free_to_target = 0.93 * state.drive_total_gb(drive) - state.drive_used_gb(drive)
            # fill_minutes — за скільки хвилин дійти до ~93 % (за замовчуванням —
            # уся тривалість інциденту); у ланцюжку з бекапом — коротше
            fill_minutes = params.get("fill_minutes") or params["duration_min"]
            params.update({
                "folder": folder,
                "drive": drive,
                "fill_minutes": fill_minutes,
                "target_gb": max(free_to_target, 1),
                "rate_gb_per_min": max(free_to_target, 1) / max(fill_minutes, 1),
                "added_gb": 0.0,
                "warned": False,
            })
        # Росте, поки не досягне цілі, далі тримається на ~93 % до «прибирання»
        step = min(params["rate_gb_per_min"] * rng.uniform(0.7, 1.3),
                   max(params["target_gb"] - params["added_gb"], 0))
        state.folder_gb[params["folder"]] += step
        params["added_gb"] += step
        values["disk_write_mbs"] += step * 1024 / 60   # ГБ/хв -> МБ/с

        events = []
        if state.drive_used_pct(params["drive"]) > 90 and not params["warned"]:
            params["warned"] = True
            events.append(event(
                "System", "srv", 2013, "Warning",
                f"The {params['drive']} disk is at or near capacity. "
                "You may need to delete some files."))
        return events

    def end(self, state, params, rng):
        # «Прибирання»: прибираємо все, що наросло під час інциденту
        folder = params.get("folder")
        if folder:
            state.folder_gb[folder] = max(state.folder_gb[folder] - params["added_gb"], 0.1)
        return []


class ServiceStop(Scenario):
    """Служба несподівано зупинилась (подія 7034), потім її запустили знову."""
    name = "service_stop"
    duration_min = (5, 30)

    def start(self, state, params, rng):
        service = self.target(state, self.name)["service"]
        params["service"] = service
        state.service_status[service] = "stopped"
        return [event(
            "System", "Service Control Manager", 7034, "Error",
            f"The {service} service terminated unexpectedly. It has done this 1 time(s).")]

    def end(self, state, params, rng):
        service = params["service"]
        state.service_status[service] = "running"
        state.restart_target({"service": service}, rng)
        return [event(
            "System", "Service Control Manager", 7036, "Information",
            f"The {service} service entered the running state.")]


class ApppoolCrash(Scenario):
    """Робочий процес пулу падає двічі поспіль, WAS вимикає пул (5002).

    Пул — з params["pool"] (для ланцюжків) або випадковий зі scenario_targets.
    """
    name = "apppool_crash"
    duration_min = (5, 20)

    def start(self, state, params, rng):
        if not params.get("pool"):
            params["pool"] = rng.choice(self.target(state, self.name)["pools"])
        return []

    def apply(self, state, values, minute, params, rng):
        pool = params["pool"]
        target = {"process": "w3wp.exe", "app_pool": pool}
        events = []
        if minute in (0, 1):
            # Процес падає (1000), WAS це фіксує (5009); пул ще пробує перезапуск
            pid = state.pid_for_target(target)
            events.append(event(
                "Application", "Application Error", 1000, "Error",
                "Faulting application name: w3wp.exe, Faulting module name: AppModule.dll, "
                f"Exception code: 0xc0000005, Faulting process id: {pid}"))
            events.append(event(
                "System", "WAS", 5009, "Warning",
                f"A process serving application pool '{pool}' terminated unexpectedly. "
                f"The process id was '{pid}'. The process exit code was '0xc0000005'."))
            state.restart_target(target, rng)   # наступна спроба — новий PID
        if minute == 2:
            state.service_status[pool] = "stopped"
            events.append(event(
                "System", "WAS", 5002, "Error",
                f"Application pool '{pool}' is being automatically disabled due to a series "
                "of failures in the process(es) serving that application pool."))

        # Процесу пулу немає; сайт на цьому пулі відповідає 503
        values["process_overrides"].append({"name": "w3wp.exe", "app_pool": pool, "hide": True})
        http = values.get("http")
        if http and state.profile["checks"]["http"].get("app_pool") == pool:
            # IIS одразу відповідає 503 — швидко, але з помилкою
            http["code"] = 503
            http["response_ms"] = rng.uniform(5, 30)
            values["app_response_ms"] = http["response_ms"]
            values["app_error_pct"] = rng.uniform(90, 100)
        elif values.get("app_error_pct") is not None:
            values["app_error_pct"] = rng.uniform(20, 40)   # падає лише частина запитів (API)
        return events

    def end(self, state, params, rng):
        pool = params["pool"]
        state.service_status[pool] = "running"
        state.restart_target({"process": "w3wp.exe", "app_pool": pool}, rng)
        return []


class SiteDown(Scenario):
    """Сайт недоступний, хоча пули працюють (наприклад, застосунок втратив зв'язок з БД).

    Варіанти:
        error — повна недоступність: кожен запит отримує 500, помилок 90–100 %;
        slow  — сторінки відкриваються, але довше 2 с: перевірки падають
                за таймаутом, а помилок мало (2–5 %).
    """
    name = "site_down"
    duration_min = (5, 30)

    def start(self, state, params, rng):
        if not params.get("variant"):
            params["variant"] = rng.choice(["error", "slow"])
        return []

    def apply(self, state, values, minute, params, rng):
        http = values.get("http")
        if params["variant"] == "error":
            values["app_response_ms"] = rng.uniform(50, 300)   # помилка повертається швидко
            values["app_error_pct"] = rng.uniform(90, 100)
            if http:
                http["code"] = 500
                http["response_ms"] = values["app_response_ms"]
        else:
            values["app_response_ms"] = rng.uniform(2500, 6000)
            values["app_error_pct"] = rng.uniform(2, 5)
            if http:
                http["code"] = 200
                http["response_ms"] = values["app_response_ms"]
        return []


class BackupStale(Scenario):
    """Бекап не вдався (3041); свіжого бекапу немає.

    Історія: інцидент починається рівно о 02:00 — нічний бекап не запускається,
             вік бекапу природно переходить за 24 год; закінчується з наступним
             успішним бекапом.
    Живий режим: діє одразу — подія 3041 зараз, вік бекапу одразу > 24 год;
             після завершення — новий успішний бекап.
    """
    name = "backup_stale"
    duration_min = (24 * 60 + 40, 24 * 60 + 40)

    def start(self, state, params, rng):
        if params.get("reason") == "no_space":
            # Ланцюжок «диск → бекап»: диск уже заповнений, нічний бекап
            # не поміститься і впаде сам (події 18210 і 3041 пише simulator.py).
            # Сценарій лише позначає інцидент у «ключі відповідей»
            return []
        state.backup_fail_next = True    # найближчий нічний бекап не запуститься
        state.last_backup_ok = False
        if params.get("live") and state.last_backup_end is not None:
            # Робимо вигляд, що останній успішний бекап був понад добу тому
            state.last_backup_end = min(state.last_backup_end,
                                        state.now - timedelta(hours=25))
        database = state.profile["backup"]["database"]
        return [event(
            "Application", "MSSQLSERVER", 3041, "Error",
            f"BACKUP failed to complete the command BACKUP DATABASE {database}. "
            "Check the backup application log for detailed messages.")]

    def end(self, state, params, rng):
        if params.get("live"):
            state.backup_fail_next = False
            return state.complete_backup()   # «адміністратор запустив бекап вручну»
        return []   # в історії успішний бекап робить звичайний нічний розклад


class CertExpiring(Scenario):
    """Сертифікат сайту наближається до кінця строку, потім його оновлюють.

    Історія: строк заздалегідь підібрано так, що на початку інциденту він
             опускається нижче 30 днів і далі природно зменшується на 1 на добу.
    Живий режим: строк прискорено падає з ~30 до ~5 днів за тривалість сценарію.
    """
    name = "cert_expiring"
    duration_min = (3 * 24 * 60, 5 * 24 * 60)

    def start(self, state, params, rng):
        if state.cert_days is not None:
            state.cert_days = min(state.cert_days, 29.9)
        return []

    def apply(self, state, values, minute, params, rng):
        if params.get("live") and state.cert_days is not None:
            state.cert_days = 29.9 - 25 * minute / max(params["duration_min"], 1)
        return []

    def end(self, state, params, rng):
        state.cert_days = 365.0   # «оновили» сертифікат
        return []


# Усі сценарії за назвою
SCENARIOS: dict[str, Scenario] = {
    s.name: s for s in (
        CpuSpike(), MemoryLeak(), DiskFill(), ServiceStop(),
        ApppoolCrash(), SiteDown(), BackupStale(), CertExpiring(),
    )
}
